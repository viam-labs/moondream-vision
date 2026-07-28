from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from src.moondream import moondream as Moondream


def make_config(attrs: dict, name: str = "moondream") -> ComponentConfig:
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def make_jpeg_image(width: int = 100, height: int = 50) -> ViamImage:
    buf = BytesIO()
    Image.new("RGB", (width, height), color="red").save(buf, format="JPEG")
    return ViamImage(data=buf.getvalue(), mime_type=CameraMimeType.JPEG)


def make_camera(image: ViamImage | None = None) -> MagicMock:
    cam = MagicMock(spec=Camera)
    cam.get_images = AsyncMock(return_value=([image or make_jpeg_image()], None))
    return cam


@pytest.fixture
def mock_vl():
    model = MagicMock()
    with patch("src.moondream.md.vl", return_value=model) as vl:
        yield vl, model


@pytest.fixture
def service(mock_vl):
    _, model = mock_vl
    cam = make_camera()
    deps = {Camera.get_resource_name("cam"): cam}
    config = make_config({"api_key": "test-key", "camera": "cam"})
    instance = Moondream.new(config, deps)
    instance._test_camera = cam
    instance._test_model = model
    return instance


class TestValidate:
    def test_requires_api_key(self):
        with pytest.raises(Exception, match="api_key is required"):
            Moondream.validate(make_config({"camera": "cam"}))

    def test_requires_camera(self):
        with pytest.raises(Exception, match="camera is required"):
            Moondream.validate(make_config({"api_key": "test-key"}))

    def test_returns_camera_dependency(self):
        assert Moondream.validate(make_config({"api_key": "test-key", "camera": "cam"})) == (
            ["cam"],
            [],
        )

    def test_accepts_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MOONDREAM_API_KEY", "env-key")
        assert Moondream.validate(make_config({"camera": "cam"})) == (["cam"], [])


class TestReconfigure:
    def test_defaults_to_local(self, mock_vl):
        vl, _ = mock_vl
        cam = make_camera()
        Moondream.new(
            make_config({"api_key": "test-key", "camera": "cam"}),
            {Camera.get_resource_name("cam"): cam},
        )
        vl.assert_called_once_with(api_key="test-key", local=True)

    def test_cloud_when_local_false(self, mock_vl):
        vl, _ = mock_vl
        cam = make_camera()
        Moondream.new(
            make_config({"api_key": "test-key", "camera": "cam", "local": False}),
            {Camera.get_resource_name("cam"): cam},
        )
        vl.assert_called_once_with(api_key="test-key", local=False)

    def test_passes_model(self, mock_vl):
        vl, _ = mock_vl
        cam = make_camera()
        Moondream.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "model": "moondream2",
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )
        vl.assert_called_once_with(api_key="test-key", local=True, model="moondream2")

    def test_uses_env_api_key(self, mock_vl, monkeypatch):
        vl, _ = mock_vl
        monkeypatch.setenv("MOONDREAM_API_KEY", "env-key")
        cam = make_camera()
        Moondream.new(
            make_config({"camera": "cam"}),
            {Camera.get_resource_name("cam"): cam},
        )
        vl.assert_called_once_with(api_key="env-key", local=True)


class TestClassifications:
    @pytest.mark.asyncio
    async def test_default_question(self, service):
        service._test_model.query.return_value = {"answer": "a red square"}
        result = await service.get_classifications(make_jpeg_image(), 1)
        assert result == [{"class_name": "a red square", "confidence": 1}]
        args, kwargs = service._test_model.query.call_args
        assert args[1] == "describe this image"
        assert kwargs.get("reasoning") is False

    @pytest.mark.asyncio
    async def test_config_classification_prompt(self, mock_vl):
        _, model = mock_vl
        model.query.return_value = {"answer": "a helmet"}
        cam = make_camera()
        service = Moondream.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "classification_prompt": "what safety gear is visible?",
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )

        await service.get_classifications(make_jpeg_image(), 1)

        assert model.query.call_args[0][1] == "what safety gear is visible?"

    @pytest.mark.asyncio
    async def test_extra_question_overrides_config_prompt(self, mock_vl):
        _, model = mock_vl
        model.query.return_value = {"answer": "yes"}
        cam = make_camera()
        service = Moondream.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "classification_prompt": "what safety gear is visible?",
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )

        result = await service.get_classifications(
            make_jpeg_image(), 1, extra={"question": "is there a person?"}
        )

        assert result[0]["class_name"] == "yes"
        assert model.query.call_args[0][1] == "is there a person?"

    @pytest.mark.asyncio
    async def test_config_reasoning(self, mock_vl):
        _, model = mock_vl
        model.query.return_value = {"answer": "detailed"}
        cam = make_camera()
        service = Moondream.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "reasoning": True,
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )

        await service.get_classifications(make_jpeg_image(), 1)

        assert model.query.call_args.kwargs.get("reasoning") is True

    @pytest.mark.asyncio
    async def test_extra_reasoning_overrides_config(self, mock_vl):
        _, model = mock_vl
        model.query.return_value = {"answer": "quick"}
        cam = make_camera()
        service = Moondream.new(
            make_config(
                {
                    "api_key": "test-key",
                    "camera": "cam",
                    "reasoning": True,
                }
            ),
            {Camera.get_resource_name("cam"): cam},
        )

        await service.get_classifications(
            make_jpeg_image(), 1, extra={"reasoning": False}
        )

        assert model.query.call_args.kwargs.get("reasoning") is False

    @pytest.mark.asyncio
    async def test_custom_question(self, service):
        service._test_model.query.return_value = {"answer": "yes"}
        result = await service.get_classifications(
            make_jpeg_image(), 1, extra={"question": "is there a person?"}
        )
        assert result[0]["class_name"] == "yes"
        assert service._test_model.query.call_args[0][1] == "is there a person?"

    @pytest.mark.asyncio
    async def test_from_camera(self, service):
        service._test_model.query.return_value = {"answer": "from cam"}
        result = await service.get_classifications_from_camera("cam", 1)
        assert result[0]["class_name"] == "from cam"
        service._test_camera.get_images.assert_awaited()

    @pytest.mark.asyncio
    async def test_from_camera_uses_requested_camera(self, mock_vl):
        _, model = mock_vl
        model.query.return_value = {"answer": "from cam-b"}
        cam_a = make_camera()
        cam_b = make_camera()
        deps = {
            Camera.get_resource_name("cam-a"): cam_a,
            Camera.get_resource_name("cam-b"): cam_b,
        }
        service = Moondream.new(
            make_config({"api_key": "test-key", "camera": "cam-a"}),
            deps,
        )

        await service.get_classifications_from_camera("cam-b", 1)

        cam_b.get_images.assert_awaited()
        cam_a.get_images.assert_not_awaited()


class TestDetections:
    @pytest.mark.asyncio
    async def test_auto_label_all_objects(self, service):
        service._test_model.query.return_value = {"answer": "person, chair"}
        service._test_model.detect.side_effect = [
            {"objects": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]},
            {"objects": [{"x_min": 0.5, "y_min": 0.5, "x_max": 0.9, "y_max": 0.9}]},
        ]

        result = await service.get_detections(make_jpeg_image(100, 50))

        prompt = service._test_model.query.call_args[0][1]
        assert "List all the objects you can see" in prompt
        assert service._test_model.query.call_args.kwargs.get("reasoning") is False
        assert service._test_model.detect.call_count == 2
        assert [d["class_name"] for d in result] == ["person", "chair"]
        assert result[0]["x_min"] == 10
        assert result[0]["y_min"] == 10
        assert result[0]["x_max"] == 30
        assert result[0]["y_max"] == 20
        assert result[0]["x_min_normalized"] == pytest.approx(0.1)
        assert result[1]["x_min"] == 50
        assert result[1]["y_max"] == 45

    @pytest.mark.asyncio
    async def test_query_limits_object_list(self, service):
        service._test_model.query.return_value = {"answer": "person"}
        service._test_model.detect.return_value = {
            "objects": [{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}]
        }

        result = await service.get_detections(
            make_jpeg_image(), extra={"query": "people"}
        )

        prompt = service._test_model.query.call_args[0][1]
        assert "List all people you can see" in prompt
        service._test_model.detect.assert_called_once()
        assert result[0]["class_name"] == "person"

    @pytest.mark.asyncio
    async def test_detection_query_respects_reasoning_extra(self, service):
        service._test_model.query.return_value = {"answer": "person"}
        service._test_model.detect.return_value = {
            "objects": [{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}]
        }

        await service.get_detections(make_jpeg_image(), extra={"reasoning": True})

        assert service._test_model.query.call_args.kwargs.get("reasoning") is True

    @pytest.mark.asyncio
    async def test_objects_extra_skips_listing_query(self, service):
        service._test_model.detect.return_value = {
            "objects": [{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}]
        }

        result = await service.get_detections(
            make_jpeg_image(), extra={"objects": "cup, bowl"}
        )

        assert [d["class_name"] for d in result] == ["cup", "bowl"]
        service._test_model.query.assert_not_called()
        assert service._test_model.detect.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_object_list(self, service):
        service._test_model.query.return_value = {"answer": ""}
        result = await service.get_detections(make_jpeg_image())
        assert result == []
        service._test_model.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_from_camera(self, service):
        service._test_model.query.return_value = {"answer": "cup"}
        service._test_model.detect.return_value = {
            "objects": [{"x_min": 0, "y_min": 0, "x_max": 0.5, "y_max": 0.5}]
        }
        result = await service.get_detections_from_camera("cam")
        assert len(result) == 1
        service._test_camera.get_images.assert_awaited()

    @pytest.mark.asyncio
    async def test_from_camera_uses_requested_camera(self, mock_vl):
        _, model = mock_vl
        model.query.return_value = {"answer": "cup"}
        model.detect.return_value = {
            "objects": [{"x_min": 0, "y_min": 0, "x_max": 0.5, "y_max": 0.5}]
        }
        cam_a = make_camera()
        cam_b = make_camera()
        service = Moondream.new(
            make_config({"api_key": "test-key", "camera": "cam-a"}),
            {
                Camera.get_resource_name("cam-a"): cam_a,
                Camera.get_resource_name("cam-b"): cam_b,
            },
        )

        await service.get_detections_from_camera("cam-b")

        cam_b.get_images.assert_awaited()
        cam_a.get_images.assert_not_awaited()


class TestPropertiesAndCaptureAll:
    @pytest.mark.asyncio
    async def test_properties(self, service):
        props = await service.get_properties()
        assert props.classifications_supported is True
        assert props.detections_supported is True
        assert props.object_point_clouds_supported is False

    @pytest.mark.asyncio
    async def test_capture_all_respects_flags(self, service):
        service._test_model.query.return_value = {"answer": "person, chair"}
        service._test_model.detect.side_effect = [
            {"objects": [{"x_min": 0, "y_min": 0, "x_max": 0.5, "y_max": 0.5}]},
            {"objects": [{"x_min": 0.5, "y_min": 0.5, "x_max": 1, "y_max": 1}]},
        ]

        result = await service.capture_all_from_camera(
            "cam",
            return_image=True,
            return_classifications=True,
            return_detections=True,
        )

        assert result.image is not None
        assert result.classifications[0]["class_name"] == "person, chair"
        assert [d["class_name"] for d in result.detections] == ["person", "chair"]
        # One classification query; detection reuses that text instead of querying again
        assert service._test_model.query.call_count == 1
        assert service._test_model.detect.call_count == 2

    @pytest.mark.asyncio
    async def test_capture_all_detections_only_still_lists_objects(self, service):
        service._test_model.query.return_value = {"answer": "box"}
        service._test_model.detect.return_value = {
            "objects": [{"x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}]
        }

        result = await service.capture_all_from_camera(
            "cam",
            return_detections=True,
        )

        assert result.detections[0]["class_name"] == "box"
        assert service._test_model.query.call_count == 1

    @pytest.mark.asyncio
    async def test_capture_all_skips_unrequested(self, service):
        result = await service.capture_all_from_camera("cam")
        assert result.image is not None
        assert not result.classifications
        assert not result.detections
        service._test_model.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_all_uses_requested_camera(self, mock_vl):
        _, model = mock_vl
        cam_a = make_camera()
        cam_b = make_camera()
        service = Moondream.new(
            make_config({"api_key": "test-key", "camera": "cam-a"}),
            {
                Camera.get_resource_name("cam-a"): cam_a,
                Camera.get_resource_name("cam-b"): cam_b,
            },
        )

        await service.capture_all_from_camera("cam-b")

        cam_b.get_images.assert_awaited()
        cam_a.get_images.assert_not_awaited()
