from typing import ClassVar, Mapping, Optional, Any, List, cast
from typing_extensions import Self
import os

from viam.proto.common import PointCloudObject
from viam.proto.service.vision import Classification, Detection
from viam.utils import ValueTypes

from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

from viam.services.vision import Vision, CaptureAllResult
from viam.proto.service.vision import GetPropertiesResponse
from viam.components.camera import Camera, ViamImage
from viam.media.utils.pil import viam_to_pil_image
from viam.media.video import CameraMimeType
from viam.logging import getLogger

import moondream as md

LOGGER = getLogger(__name__)

DEFAULT_CLASSIFICATION_PROMPT = "describe this image"

class moondream(Vision, Reconfigurable):
    
    """
    Vision represents a Vision service.
    """
    

    MODEL: ClassVar[Model] = Model(ModelFamily("viam-labs", "vision"), "moondream")
    
    model: Any
    DEPS: Mapping[ResourceName, ResourceBase]
    classification_prompt: str
    reasoning: bool

    # Constructor
    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        my_class = cls(config.name)
        my_class.reconfigure(config, dependencies)
        return my_class

    # Validates JSON Configuration
    @classmethod
    def validate(cls, config: ComponentConfig):
        fields = config.attributes.fields
        api_key = fields["api_key"].string_value or os.environ.get("MOONDREAM_API_KEY", "")
        if not api_key:
            raise Exception("api_key is required (set attributes.api_key or MOONDREAM_API_KEY)")
        camera = fields["camera"].string_value
        if not camera:
            raise Exception("camera is required")
        return [camera], []

    # Handles attribute reconfiguration
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        self.DEPS = dependencies
        fields = config.attributes.fields

        api_key = fields["api_key"].string_value or os.environ.get("MOONDREAM_API_KEY", "")
        if not api_key:
            raise Exception("api_key is required (set attributes.api_key or MOONDREAM_API_KEY)")

        # Ensure the configured camera dependency is present
        camera_name = fields["camera"].string_value
        if Camera.get_resource_name(camera_name) not in dependencies:
            raise Exception(f"camera dependency '{camera_name}' not found")

        self.classification_prompt = (
            fields["classification_prompt"].string_value or DEFAULT_CLASSIFICATION_PROMPT
        )

        # Default false; set reasoning=true for higher-quality answers (more latency)
        self.reasoning = False
        if "reasoning" in fields:
            self.reasoning = fields["reasoning"].bool_value

        # Default to local Photon inference; set local=false for Moondream Cloud
        local = True
        if "local" in fields:
            local = fields["local"].bool_value

        kwargs = {"api_key": api_key, "local": local}
        model_name = fields["model"].string_value
        if model_name:
            kwargs["model"] = model_name

        mode = "local (Photon)" if local else "cloud"
        LOGGER.info(f"initializing Moondream in {mode} mode")
        self.model = md.vl(**kwargs)
        return
    
    async def get_cam_image(self, camera_name: str) -> ViamImage:
        cam = cast(Camera, self.DEPS[Camera.get_resource_name(camera_name)])
        images, _ = await cam.get_images()
        if not images:
            raise Exception("get_images from cam returned no images")
        for img in images:
            if img.mime_type == CameraMimeType.JPEG:
                return img
        raise Exception(f"no images from cam is {CameraMimeType.JPEG}")

    def _resolve_reasoning(self, extra: Optional[Mapping[str, Any]] = None) -> bool:
        if extra is not None and "reasoning" in extra:
            return bool(extra["reasoning"])
        return self.reasoning

    def _object_names_from_text(self, text: str) -> List[str]:
        return [obj.strip() for obj in str(text).split(",") if obj.strip()]

    def _object_list_prompt(self, query: Optional[str] = None) -> str:
        if query and str(query).strip():
            return (
                f"List all {str(query).strip()} you can see in this image. "
                "Return your answer as a simple comma-separated list of object names."
            )
        return (
            "List all the objects you can see in this image. "
            "Return your answer as a simple comma-separated list of object names."
        )

    def _query_object_list(
        self,
        pil_image,
        query: Optional[str] = None,
        *,
        reasoning: bool = False,
    ) -> tuple:
        """Return (raw_answer, object_names) from the detection listing prompt."""
        answer = (
            self.model.query(
                pil_image, self._object_list_prompt(query), reasoning=reasoning
            )["answer"]
            or ""
        )
        return answer, self._object_names_from_text(answer)

    def _list_objects(
        self,
        pil_image,
        query: Optional[str] = None,
        *,
        reasoning: bool = False,
    ) -> List[str]:
        """Query Moondream for a comma-separated list of objects to detect."""
        _, object_names = self._query_object_list(
            pil_image, query, reasoning=reasoning
        )
        return object_names

    def _detect_objects(self, pil_image, object_names: List[str]) -> List[Detection]:
        """Run detect for each object name and convert to Viam detections."""
        width, height = pil_image.size
        detections = []
        for object_name in object_names:
            result = self.model.detect(pil_image, object_name)
            for obj in result.get("objects") or []:
                x_min_n = float(obj["x_min"])
                y_min_n = float(obj["y_min"])
                x_max_n = float(obj["x_max"])
                y_max_n = float(obj["y_max"])
                detections.append({
                    "x_min": int(x_min_n * width),
                    "y_min": int(y_min_n * height),
                    "x_max": int(x_max_n * width),
                    "y_max": int(y_max_n * height),
                    "x_min_normalized": x_min_n,
                    "y_min_normalized": y_min_n,
                    "x_max_normalized": x_max_n,
                    "y_max_normalized": y_max_n,
                    "confidence": 1,
                    "class_name": object_name,
                })
        return detections
    
    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None
    ) -> List[Detection]:
        return await self.get_detections(await self.get_cam_image(camera_name), extra=extra)

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        # Auto-label: query for object names, then detect each.
        # Pass extra={"query": "..."} to limit which objects are listed.
        # Pass extra={"objects": "a, b"} or a list to skip the listing query.
        # See https://docs.moondream.ai/sample-projects/automatic-detection-labeling
        pil_image = viam_to_pil_image(image)
        object_names: List[str] = []
        if extra is not None and extra.get("objects") is not None:
            objects = extra["objects"]
            if isinstance(objects, str):
                object_names = self._object_names_from_text(objects)
            else:
                object_names = [str(obj).strip() for obj in objects if str(obj).strip()]
        else:
            query = extra.get("query") if extra else None
            object_names = self._list_objects(
                pil_image, query, reasoning=self._resolve_reasoning(extra)
            )
        return self._detect_objects(pil_image, object_names)
    
    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        return await self.get_classifications(await self.get_cam_image(camera_name), count, extra=extra)

    
    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        classifications = []
        question = self.classification_prompt
        if extra is not None and extra.get("question") is not None:
            question = extra["question"]
        result = self.model.query(
            viam_to_pil_image(image),
            question,
            reasoning=self._resolve_reasoning(extra),
        )["answer"]
        classifications.append({"class_name": result, "confidence": 1})
        return classifications

    
    async def get_object_point_clouds(
        self, camera_name: str, *, extra: Optional[Mapping[str, Any]] = None, timeout: Optional[float] = None
    ) -> List[PointCloudObject]:
        return
    
    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None) -> Mapping[str, ValueTypes]:
        return

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> CaptureAllResult:
        result = CaptureAllResult()
        result.image = await self.get_cam_image(camera_name)

        # Classifications alone use classification_prompt (often a description).
        # Detections alone use the object-listing prompt, then detect.
        # When both are requested, share one listing query so we stay fast and
        # classifications are object names rather than descriptive sentences.
        if return_classifications and return_detections:
            pil_image = viam_to_pil_image(result.image)
            extra = extra or {}
            if extra.get("objects") is not None:
                objects = extra["objects"]
                if isinstance(objects, str):
                    answer = objects
                    object_names = self._object_names_from_text(objects)
                else:
                    object_names = [
                        str(obj).strip() for obj in objects if str(obj).strip()
                    ]
                    answer = ", ".join(object_names)
            else:
                answer, object_names = self._query_object_list(
                    pil_image,
                    extra.get("query"),
                    reasoning=self._resolve_reasoning(extra),
                )
            result.classifications = [{"class_name": answer, "confidence": 1}]
            result.detections = self._detect_objects(pil_image, object_names)
        else:
            if return_classifications:
                result.classifications = await self.get_classifications(
                    result.image, 1, extra=extra
                )
            if return_detections:
                result.detections = await self.get_detections(
                    result.image, extra=extra
                )
        return result

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=True,
            object_point_clouds_supported=False
            )
