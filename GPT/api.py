from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
import base64
import mimetypes
import os

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY", "EMPTY"),
    base_url=os.getenv("QWEN_BASE_URL", "http://10.79.128.145:8001/v1"),
)


@retry(
    retry=retry_if_exception_type(
        (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    ),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
)
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def gpt_infer(system, text, image_list, model="gpt-4-vision-preview", max_tokens=600,
              response_format=None, defined_indice=None, json_schema=None):

    user_content = []
    for i, image in enumerate(image_list):
        if image is not None:
            
            if defined_indice:
                j = defined_indice[i]
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Image {j}:"
                    },
                )
            else:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Image {i}:"
                    },
                )
            with open(image, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = mimetypes.guess_type(image)[0] or "image/jpeg"

            image_message = {
                     "type": "image_url",
                     "image_url": {
                         "url": f"data:{mime_type};base64,{image_base64}",
                     }
                 }
            user_content.append(image_message)

    user_content.append(
        {
            "type": "text",
            "text": text
        }
    )

    messages = [
        {"role": "system",
         "content": system
         },
        {"role": "user",
         "content": user_content
         }
    ]
    # use the same version of 4o model with previous zero-shot model for agent performance comparision
    # can be changed to other cheaper versions of 4o model for cost saving, but with slower api response 
    # if model is gpt-4o, specify use the model of gpt-4o-2024-05-13
    if model == "gpt-4o":
        model="gpt-4o-2024-05-13" 
    
    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if json_schema is not None:
        request_kwargs["extra_body"] = {
            "structured_outputs": {
                "json": json_schema,
            }
        }
    elif response_format:
        request_kwargs["response_format"] = response_format

    chat_message = completion_with_backoff(**request_kwargs)

    #print('gpt chat_message:', chat_message)
    answer = chat_message.choices[0].message.content
    tokens = chat_message.usage

    return answer, tokens
