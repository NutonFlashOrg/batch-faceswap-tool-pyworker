# ComfyUI Wan 2.2 PyWorker

This is the PyWorker implementation for running **Wan 2.2 T2V A14B** text-to-video workflows in ComfyUI. It provides a unified interface for executing complete ComfyUI video-generation workflows through a proxy-based architecture and returning generated video assets.

Each request has a static cost of `10000`. ComfyUI does not support concurrent workloads, and there is no provision to run multiple ComfyUI instances per worker node.

## Requirements

This worker requires the following components:

- ComfyUI (https://github.com/comfyanonymous/ComfyUI)
- ComfyUI API Wrapper (https://github.com/ai-dock/comfyui-api-wrapper)
- Wan 2.2 T2V A14B models and required custom nodes

A Docker image is provided with all required Wan 2.2 models pre-installed, but any image may be used if the above requirements are met.

## Endpoint

The worker exposes a single synchronous endpoint:

- `/generate/sync`: Processes a complete ComfyUI workflow JSON and generates video output

## Request Format

The Wan 2.2 worker **only supports custom workflow mode**. Modifier-based workflows are not supported.

```json
{
  "input": {
    "request_id": "uuid-string",
    "workflow_json": {
      // Complete ComfyUI Wan 2.2 workflow JSON
    },
    "s3": { },
    "webhook": { }
  }
}
```

## Request Fields

### Required Fields

- `input`: Container for all request parameters
- `input.workflow_json`: Complete ComfyUI workflow graph for Wan 2.2 video generation

### Optional Fields

- `input.request_id`: Client-defined request identifier
- `input.s3`: S3-compatible storage configuration
- `input.webhook`: Webhook configuration for completion notifications

The special string `"__RANDOM_INT__"` may be used in the workflow JSON and will be replaced with a random integer before submission to ComfyUI.

## S3 Configuration

Generated video assets can be automatically uploaded to S3-compatible storage. Configuration can be supplied per request or via environment variables. Request-level values take precedence.

### Via Request JSON

```json
"s3": {
  "access_key_id": "your-s3-access-key",
  "secret_access_key": "your-s3-secret-access-key",
  "endpoint_url": "https://s3.amazonaws.com",
  "bucket_name": "your-bucket",
  "region": "us-east-1"
}
```

### Via Environment Variables

```bash
S3_ACCESS_KEY_ID=your-key
S3_SECRET_ACCESS_KEY=your-secret
S3_BUCKET_NAME=your-bucket
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_REGION=us-east-1
```

## Webhook Configuration

Webhooks are triggered on request completion or failure.

### Via Request JSON

```json
"webhook": {
  "url": "https://your-webhook-url",
  "extra_params": {
    "custom_field": "value"
  }
}
```

### Via Environment Variables

```bash
WEBHOOK_URL=https://your-webhook-url
WEBHOOK_TIMEOUT=30
```

## Example Request

### Wan 2.2 Text-to-Video Workflow

```json
{
  "input": {
    "workflow_json": {
      "90": {
        "inputs": {
          "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
          "type": "wan",
          "device": "default"
        },
        "class_type": "CLIPLoader"
      },
      "99": {
        "inputs": {
          "text": "A cinematic slow-motion portrait of a woman turning her head",
          "clip": ["90", 0]
        },
        "class_type": "CLIPTextEncode"
      },
      "104": {
        "inputs": {
          "width": 640,
          "height": 640,
          "length": 81,
          "batch_size": 1
        },
        "class_type": "EmptyHunyuanLatentVideo"
      }
    }
  }
}
```

## Response Format

A successful response includes execution metadata, ComfyUI output details, and generated video assets.

### Response Fields

- `id`: Unique request identifier
- `status`: `completed`, `failed`, `processing`, `generating`, or `queued`
- `message`: Human-readable status message
- `comfyui_response`: Raw response from ComfyUI, including execution status and progress
- `output`: Array of generated outputs
- `timings`: Timing information for the request

### Output Object

Each entry in `output` includes:

- `filename`: Generated file name (e.g., `.mp4`)
- `local_path`: File path on the worker
- `url`: Pre-signed download URL (if S3 is configured)
- `type`: Output type (`output`)
- `subfolder`: Output directory (e.g., `video`)
- `node_id`: ComfyUI node that produced the output
- `output_type`: Output category (e.g., `images`)

## Notes and Limitations

- Only full ComfyUI workflow JSONs are supported
- Concurrent requests are not supported per worker
- Wan 2.2 models must be installed before processing requests
- Video generation workflows may take several minutes depending on resolution, length, and GPU performance