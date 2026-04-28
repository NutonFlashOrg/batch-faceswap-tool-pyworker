import logging
import json
import os
import sys
import argparse

from vastai import Serverless
import asyncio

# ---------------------- Logging ----------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

# ---------------------- Defaults ----------------------
DEFAULT_PROMPT = "Think step by step: Tell me about the Python programming language."

ENDPOINT_NAME = "TGI-Prod2"       # change this to your TGI endpoint name
MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7


# ---------------------- API Calls ----------------------
async def call_generate(client: Serverless, *, endpoint_name: str, prompt: str, **kwargs) -> dict:
    """Non-streaming generation via /generate endpoint"""
    endpoint = await client.get_endpoint(name=endpoint_name)

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": kwargs.get("max_tokens", MAX_TOKENS),
            "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "return_full_text": False,
        }
    }
    log.debug("POST /generate %s", json.dumps(payload)[:500])
    resp = await endpoint.request("/generate", payload, cost=payload["parameters"]["max_new_tokens"])
    return resp["response"]


async def call_generate_stream(client: Serverless, *, endpoint_name: str, prompt: str, **kwargs):
    """Streaming generation via /generate_stream endpoint"""
    endpoint = await client.get_endpoint(name=endpoint_name)

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": kwargs.get("max_tokens", MAX_TOKENS),
            "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
            "do_sample": True,
            "return_full_text": False,
        }
    }
    log.debug("STREAM /generate_stream %s", json.dumps(payload)[:500])
    resp = await endpoint.request(
        "/generate_stream",
        payload,
        cost=payload["parameters"]["max_new_tokens"],
        stream=True,
    )
    return resp["response"]  # async generator


# ---------------------- Demo Runner ----------------------
class APIDemo:
    """Demo and testing functionality for the TGI API client"""

    def __init__(self, client: Serverless, endpoint_name: str):
        self.client = client
        self.endpoint_name = endpoint_name

    async def handle_streaming_response(self, stream) -> str:
        """Process streaming response and print tokens"""
        full_response = ""
        printed_answer = False

        async for event in stream:
            tok = (event.get("token") or {}).get("text")
            if tok:
                if not printed_answer:
                    printed_answer = True
                    print("\n💬 Response: ", end="", flush=True)
                print(tok, end="", flush=True)
                full_response += tok

        print()  # newline
        if printed_answer:
            print(f"\nStreaming completed. Response tokens: {len(full_response.split())}")

        return full_response

    async def demo_generate(self) -> None:
        """Demo non-streaming generation"""
        print("=" * 60)
        print("GENERATE DEMO (NON-STREAMING)")
        print("=" * 60)

        response = await call_generate(
            client=self.client,
            endpoint_name=self.endpoint_name,
            prompt=DEFAULT_PROMPT,
            max_tokens=MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )
        
        print(f"\n💬 Response: {response.get('generated_text', '')}")
        print(f"\nFull Response:\n{json.dumps(response, indent=2)}")

    async def demo_generate_stream(self) -> None:
        """Demo streaming generation"""
        print("=" * 60)
        print("GENERATE DEMO (STREAMING)")
        print("=" * 60)

        stream = await call_generate_stream(
            client=self.client,
            endpoint_name=self.endpoint_name,
            prompt=DEFAULT_PROMPT,
            max_tokens=MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )

        try:
            await self.handle_streaming_response(stream)
        except Exception as e:
            log.error("\nError during streaming: %s", e, exc_info=True)

    async def interactive_chat(self) -> None:
        """Interactive session with streaming generation"""
        print("=" * 60)
        print("INTERACTIVE STREAMING SESSION")
        print("=" * 60)
        print(f"Using endpoint: {self.endpoint_name}")
        print("Type 'quit' to exit")
        print()

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    print("👋 Goodbye!")
                    break
                elif not user_input:
                    continue

                print("Assistant: ", end="", flush=True)
                stream = await call_generate_stream(
                    client=self.client,
                    endpoint_name=self.endpoint_name,
                    prompt=user_input,
                    max_tokens=MAX_TOKENS,
                    temperature=DEFAULT_TEMPERATURE,
                )

                full_response = ""
                async for event in stream:
                    tok = (event.get("token") or {}).get("text")
                    if tok:
                        print(tok, end="", flush=True)
                        full_response += tok
                print()  # newline

            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                log.error("\nError: %s", e)
                continue


# ---------------------- CLI ----------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vast TGI Demo (Serverless SDK)")
    p.add_argument("--endpoint", default=ENDPOINT_NAME, help=f"Vast endpoint name (default: {ENDPOINT_NAME})")

    modes = p.add_mutually_exclusive_group(required=False)
    modes.add_argument("--generate", action="store_true", help="Test generate endpoint (non-streaming)")
    modes.add_argument("--generate-stream", action="store_true", help="Test generate endpoint with streaming")
    modes.add_argument("--interactive", action="store_true", help="Start interactive streaming session")
    return p


async def main_async():
    args = build_arg_parser().parse_args()

    selected = sum([args.generate, args.generate_stream, args.interactive])
    if selected == 0:
        print("Please specify exactly one test mode:")
        print("  --generate        : Test generate endpoint (non-streaming)")
        print("  --generate-stream : Test generate endpoint with streaming")
        print("  --interactive     : Start interactive streaming session")
        print(f"\nExample: python {os.path.basename(sys.argv[0])} --generate-stream --endpoint my-tgi-endpoint")
        sys.exit(1)
    elif selected > 1:
        print("Please specify exactly one test mode")
        sys.exit(1)

    print("=" * 60)
    print(f"Using endpoint: {args.endpoint}")

    try:
        async with Serverless() as client:
            demo = APIDemo(client, args.endpoint)

            if args.generate:
                await demo.demo_generate()
            elif args.generate_stream:
                await demo.demo_generate_stream()
            elif args.interactive:
                await demo.interactive_chat()

    except Exception as e:
        log.error("Error during test: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
