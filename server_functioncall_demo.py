from openai import OpenAI
import json

# CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --server-port 23333 --api-keys internlm2
# api_key = "internlm2" # 但效果沒有interlm2.5

#CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2_5-1_8b-chat  --server-port 23333 --api-keys internlm2.5
# api_key = "internlm2.5"
client = OpenAI(api_key = "internlm2.5",
    base_url = "http://0.0.0.0:23333/v1")

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    print("函數調用開始：",location)
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    print(client.models.list().data[0].id)
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]
    # messages = [{"role": "user", "content": "你是誰"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        # model="gpt-4o",
        model = client.models.list().data[0].id,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    # print(response_message)
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "environment",
                    # "name": function_name,
                    "content": function_response,
                })
        second_response = client.chat.completions.create(
            model = client.models.list().data[0].id,
            messages=messages,
        )  # get a new response from the model where it can see the function response
        messages.append({
            "role":second_response.choices[0].message.role,
            "content":second_response.choices[0].message.content
        })
        return second_response, messages


current_response, all_history = run_conversation()
print("=====================")
print(current_response)
print("=====================")
print(all_history)
