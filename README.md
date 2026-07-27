# moondream modular vision service

This module implements the [rdk vision API](https://github.com/rdk/vision-api) in a viam-labs:vision:moondream model.

This model leverages [Moondream](https://docs.moondream.ai/) via the Photon inference engine to allow for image classification and querying. By default it runs locally with Photon; set `local` to `false` to use [Moondream Cloud](https://docs.moondream.ai/quickstart) instead. An API key from [moondream.ai](https://moondream.ai/c/cloud/api-keys) is required in either mode.

Local Photon inference requires an NVIDIA GPU (Ampere or newer) or an Apple Silicon Mac. See [Run Moondream Locally](https://docs.moondream.ai/running-locally) for hardware details.

## Build and Run

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/registry/configure/#add-a-modular-resource-from-the-viam-registry) and select the `viam-labs:vision:moondream` model from the [viam-labs moondream-vision module](https://app.viam.com/module/viam-labs/moondream-vision).

## Configure your vision service

> [!NOTE]  
> Before configuring your vision service, you must [create a machine](https://docs.viam.com/manage/fleet/machines/#add-a-new-machine).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/).
Click on the **Service** subtab and click **Create service**.
Select the `vision` type, then select the `viam-labs:vision:moondream` model.
Enter a name for your vision service and click **Create**.

On the new service panel, copy and paste the following attribute template into your vision service's **Attributes** box:

```json
{
  "api_key": "<your Moondream API key>",
  "camera": "<camera-name>",
  "local": true
}
```

> [!NOTE]  
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

### Attributes

The following attributes are available for `viam-labs:vision:moondream` model:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `api_key` | string | **Required** | Moondream API key from [moondream.ai](https://moondream.ai/c/cloud/api-keys). Can also be supplied via the `MOONDREAM_API_KEY` environment variable. |
| `camera` | string | **Required** | Default camera dependency for the service. Camera-based API methods use the `camera_name` argument; add extra cameras via `depends_on` if needed. |
| `local` | bool | Optional | Run with Photon locally (`true`, default) or Moondream Cloud (`false`). |
| `model` | string | Optional | Model to use. Defaults to Moondream 3 Preview. Use `"moondream2"` for Moondream 2. |

### Example Configurations

Local Photon inference (default):

```json
{
  "api_key": "YOUR_API_KEY",
  "camera": "cam"
}
```

Moondream Cloud:

```json
{
  "api_key": "YOUR_API_KEY",
  "camera": "cam",
  "local": false
}
```

Local with Moondream 2:

```json
{
  "api_key": "YOUR_API_KEY",
  "camera": "cam",
  "local": true,
  "model": "moondream2"
}
```

## API

The moondream resource provides the following methods from Viam's built-in [rdk:service:vision API](https://python.viam.dev/autoapi/viam/services/vision/client/index.html)

Camera-based methods use the `camera_name` argument. That camera must be available as a dependency (the required `camera` attribute, and any additional cameras listed in `depends_on`).

### get_classifications(image=*binary*, count)

### get_classifications_from_camera(camera_name=*string*, count)

By default, the Moondream model will be asked the question "describe this image".
If you want to ask a different question about the image, you can pass that question as the extra parameter "question".
For example:

``` python
moondream.get_classifications(image, 1, extra={"question": "what is the person wearing?"})
```

### get_detections(image=*binary*)

### get_detections_from_camera(camera_name=*string*)

Detections use Moondream's [automatic detection labeling](https://docs.moondream.ai/sample-projects/automatic-detection-labeling) flow: query the image for a comma-separated list of object names, then run detect on each name to get bounding boxes.

By default, all visible objects are listed and detected. Pass `extra={"query": "..."}` to limit the list (for example, only people or vehicles):

``` python
moondream.get_detections(image, extra={"query": "people"})
```
