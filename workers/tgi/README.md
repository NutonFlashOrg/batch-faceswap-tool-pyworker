# HuggingFace TGI PyWorker

This is the base PyWorker for HuggingFace Text Generation Inference (TGI) servers. See the [Serverless documentation](https://docs.vast.ai/serverless) for guides and how-to's.

## Instance Setup

1. Pick a template

This worker is compatible with any TGI backend. We have a template you can use or you can create your own.

- [HuggingFace TGI](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=TGI%20(Serverless))

The template can be configured via the template interface. You may want to change the model or startup arguments.

2. Follow the [getting started guide](https://docs.vast.ai/documentation/serverless/quickstart) for help with configuring your serverless setup. For testing, we recommend that you use the default options presented by the web interface.

## Client Setup (Demo)

1. Clone the PyWorker repository to your local machine and install the necessary requirements for running the test client.

```bash
git clone https://github.com/vast-ai/pyworker
cd pyworker
pip install uv
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Using the Test Client

The test client demonstrates both streaming and non-streaming generation using TGI's native API.

First, set your API key as an environment variable:

```bash
export VAST_API_KEY=<your_api_key>
```

The `--endpoint` flag is optional. If not provided, it defaults to `my-tgi-endpoint`.

### Generate (Streaming)

Call to `/generate_stream` with streaming response:

```bash
python -m workers.tgi.client --generate-stream --endpoint <ENDPOINT_NAME>
```

### Generate (Non-Streaming)

Call to `/generate` with json response:

```bash
python -m workers.tgi.client --generate --endpoint <ENDPOINT_NAME>
```

### Interactive Session (Streaming)

Interactive session with streaming responses. Type `quit` to exit.

```bash
python -m workers.tgi.client --interactive --endpoint <ENDPOINT_NAME>
```

## API Endpoints

TGI provides two primary endpoints:

### Generate (Non-Streaming)

`/generate` - Returns the complete response in a single request.

```json
{
  "inputs": "Your prompt here",
  "parameters": {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "return_full_text": false
  }
}
```

### Generate Stream (Streaming)

`/generate_stream` - Streams the response token by token.

```json
{
  "inputs": "Your prompt here",
  "parameters": {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "do_sample": true,
    "return_full_text": false
  }
}
```

## Performance Notes

The `max_new_tokens` parameter (not the prompt size) primarily impacts performance. For example, if an instance is benchmarked to process 100 tokens per second, a request with `max_new_tokens = 200` will take approximately 2 seconds to complete.
