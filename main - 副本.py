import argparse
import os
import base64
import logging
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uuid
import uvicorn
import json
from typing import List, Dict, Any, Callable, AsyncGenerator
import asyncio
import re

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 应用
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 配置
BASE_URL = "https://gemini-spawner.goutou.art"
API_KEY = "AIzaSyAMtVMc7hKS9koS6GbF8azQUD2A3qf3o5w"
MODEL = "gemini-2.0-flash-001"
SYSTEM_INSTRUCTION = "你是一个有帮助的 AI 助手，支持处理文本、图片、音频、视频和PDF等多模态输入，并可以调用函数返回结构化数据。"

# 全局变量
conversation_history: Dict[str, List[Dict[str, Any]]] = {}
clients: Dict[str, WebSocket] = {}
webui_listeners: List[Callable] = []

# 加载 tools.json
TOOLS_FILE = os.path.join(os.path.dirname(__file__), "tools.json")
TOOLS = []
if os.path.exists(TOOLS_FILE):
    try:
        with open(TOOLS_FILE, "r", encoding="utf-8") as f:
            TOOLS = json.load(f)
        logger.info(f"成功加载 tools.json，找到 {len(TOOLS)} 个工具")
    except Exception as e:
        logger.error(f"加载 tools.json 失败: {str(e)}，禁用函数调用")
        TOOLS = []
else:
    logger.warning("未找到 tools.json 文件，禁用函数调用")
    TOOLS = []

# 将文件内容编码为 base64
def encode_file_to_base64(file_content: bytes) -> str:
    return base64.b64encode(file_content).decode("utf-8")

# 下载并编码文件
async def download_and_encode_file(url: str) -> Dict[str, str]:
    logger.info(f"正在下载文件: {url}")
    async with httpx.AsyncClient(timeout=100) as client:
        response = await client.get(url)
        if response.status_code != 200:
            logger.error(f"下载失败: {url}, 状态码: {response.status_code}")
            raise HTTPException(status_code=400, detail=f"无法下载文件: {url}")
        content = response.content
        mime_type = response.headers.get("content-type", "application/octet-stream")
        return {"data": encode_file_to_base64(content), "mime_type": mime_type}

# 获取文件 MIME 类型
def get_mime_type(file_path: str) -> str:
    extension = Path(file_path).suffix.lower() if '.' in file_path else '.bin'
    mime_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".mp4": "video/mp4",
        ".mov": "video/quicktime", ".pdf": "application/pdf", ".mpeg": "audio/mpeg",
        ".aac": "audio/aac", ".flac": "audio/flac", ".m4a": "audio/mp4", ".webm": "video/webm",
        ".tiff": "image/tiff", ".bmp": "image/bmp"
    }
    return mime_types.get(extension, "application/octet-stream")

# 上传文件到谷歌 media.upload 端点
async def upload_to_gemini_media(file_content: bytes, mime_type: str) -> str:
    url = f"{BASE_URL}/upload/v1beta/files?key={API_KEY}"
    headers = {
        "Content-Type": mime_type,
        "X-Goog-Upload-Protocol": "raw"
    }
    async with httpx.AsyncClient(timeout=100) as client:
        response = await client.post(url, headers=headers, content=file_content)
        if response.status_code != 200:
            logger.error(f"文件上传失败: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=response.text)
        data = response.json()
        file_uri = data.get("file", {}).get("uri")
        if not file_uri:
            logger.error(f"未获取到 fileUri: {response.text}")
            raise HTTPException(status_code=500, detail="文件上传成功但未返回 fileUri")
        logger.info(f"文件上传成功，获取到 fileUri: {file_uri}")
        return file_uri

# 定义多模态输入类
class Text:
    def __init__(self, content: str):
        self.type = "text"
        self.content = content

    def to_dict(self):
        return {"text": self.content}

class Image:
    def __init__(self, path: str = None, url: str = None, base64: str = None, byte: bytes = None, mime_type: str = None):
        self.type = "image"
        self.source = {}
        self.url = url  # 保存 URL，延迟异步处理
        if path:
            self.source["base64"] = encode_file_to_base64(Path(path).read_bytes())
            self.source["mime_type"] = get_mime_type(path)
            self.source["filename"] = Path(path).name
        elif url:
            self.source["mime_type"] = mime_type or "image/jpeg"
            self.source["filename"] = url.split("/")[-1] or "downloaded_image"
        elif base64:
            self.source["base64"] = base64
            self.source["mime_type"] = mime_type or "image/jpeg"
            self.source["filename"] = "inline_image"
        elif byte:
            self.source["base64"] = encode_file_to_base64(byte)
            self.source["mime_type"] = mime_type or "image/jpeg"
            self.source["filename"] = "byte_image"

    async def to_dict(self):
        if self.url and "base64" not in self.source:
            data = await download_and_encode_file(self.url)
            self.source["base64"] = data["data"]
            self.source["mime_type"] = data["mime_type"]
        return {"inline_data": {"mime_type": self.source["mime_type"], "data": self.source["base64"]}}

class Audio:
    def __init__(self, path: str = None, url: str = None, base64: str = None, byte: bytes = None, mime_type: str = None):
        self.type = "audio"
        self.source = {}
        self.url = url
        if path:
            self.source["base64"] = encode_file_to_base64(Path(path).read_bytes())
            self.source["mime_type"] = get_mime_type(path)
            self.source["filename"] = Path(path).name
        elif url:
            self.source["mime_type"] = mime_type or "audio/mpeg"
            self.source["filename"] = url.split("/")[-1] or "downloaded_audio"
        elif base64:
            self.source["base64"] = base64
            self.source["mime_type"] = mime_type or "audio/mpeg"
            self.source["filename"] = "inline_audio"
        elif byte:
            self.source["base64"] = encode_file_to_base64(byte)
            self.source["mime_type"] = mime_type or "audio/mpeg"
            self.source["filename"] = "byte_audio"

    async def to_dict(self, upload: bool = False):
        if self.url and "base64" not in self.source:
            data = await download_and_encode_file(self.url)
            self.source["base64"] = data["data"]
            self.source["mime_type"] = data["mime_type"]
        if upload:
            file_uri = await upload_to_gemini_media(base64.b64decode(self.source["base64"]), self.source["mime_type"])
            return {"fileData": {"mimeType": self.source["mime_type"], "fileUri": file_uri}}
        return {"inline_data": {"mime_type": self.source["mime_type"], "data": self.source["base64"]}}

class Video:
    def __init__(self, path: str = None, url: str = None, base64: str = None, byte: bytes = None, mime_type: str = None):
        self.type = "video"
        self.source = {}
        self.url = url
        if path:
            self.source["base64"] = encode_file_to_base64(Path(path).read_bytes())
            self.source["mime_type"] = get_mime_type(path)
            self.source["filename"] = Path(path).name
        elif url:
            self.source["mime_type"] = mime_type or "video/mp4"
            self.source["filename"] = url.split("/")[-1] or "downloaded_video"
        elif base64:
            self.source["base64"] = base64
            self.source["mime_type"] = mime_type or "video/mp4"
            self.source["filename"] = "inline_video"
        elif byte:
            self.source["base64"] = encode_file_to_base64(byte)
            self.source["mime_type"] = mime_type or "video/mp4"
            self.source["filename"] = "byte_video"

    async def to_dict(self, upload: bool = False):
        if self.url and "base64" not in self.source:
            data = await download_and_encode_file(self.url)
            self.source["base64"] = data["data"]
            self.source["mime_type"] = data["mime_type"]
        if upload:
            file_uri = await upload_to_gemini_media(base64.b64decode(self.source["base64"]), self.source["mime_type"])
            return {"fileData": {"mimeType": self.source["mime_type"], "fileUri": file_uri}}
        return {"inline_data": {"mime_type": self.source["mime_type"], "data": self.source["base64"]}}

class CustomFile:
    def __init__(self, path: str = None, url: str = None, base64: str = None, byte: bytes = None, mime_type: str = None):
        self.type = "file"
        self.source = {}
        self.url = url
        if path:
            self.source["base64"] = encode_file_to_base64(Path(path).read_bytes())
            self.source["mime_type"] = get_mime_type(path)
            self.source["filename"] = Path(path).name
        elif url:
            self.source["mime_type"] = mime_type or "application/octet-stream"
            self.source["filename"] = url.split("/")[-1] or "downloaded_file"
        elif base64:
            self.source["base64"] = base64
            self.source["mime_type"] = mime_type or "application/octet-stream"
            self.source["filename"] = "inline_file"
        elif byte:
            self.source["base64"] = encode_file_to_base64(byte)
            self.source["mime_type"] = mime_type or "application/octet-stream"
            self.source["filename"] = "byte_file"

    async def to_dict(self, upload: bool = False):
        if self.url and "base64" not in self.source:
            data = await download_and_encode_file(self.url)
            self.source["base64"] = data["data"]
            self.source["mime_type"] = data["mime_type"]
        if upload and len(base64.b64decode(self.source["base64"])) > 20 * 1024 * 1024:
            file_uri = await upload_to_gemini_media(base64.b64decode(self.source["base64"]), self.source["mime_type"])
            return {"fileData": {"mimeType": self.source["mime_type"], "fileUri": file_uri}}
        return {"inline_data": {"mime_type": self.source["mime_type"], "data": self.source["base64"]}}

class Node:
    def __init__(self, messages: List[List[Any]]):  # 修改为 List[List[Any]] 支持多组消息
        self.type = "node"
        self.messages = messages  # 嵌套消息列表，每组是一个气泡

    async def to_dict(self, upload: bool = False):
        processed_messages = []
        for message_group in self.messages:  # 遍历每组消息
            group = []
            for msg in message_group:
                if hasattr(msg, 'to_dict') and asyncio.iscoroutinefunction(msg.to_dict):
                    group.append(await msg.to_dict(upload=upload))
                elif hasattr(msg, 'to_dict'):
                    group.append(msg.to_dict())
                else:
                    group.append(msg)
            processed_messages.append(group)
        return {"type": self.type, "messages": processed_messages}

class WebUIEvent:
    def __init__(self, message_list: List[Dict[str, Any]], client_id: str):
        self.client_id = client_id
        self.plain = "".join(item["content"] for item in message_list if item["type"] == "text")
        self.image = [item for item in message_list if item["type"] == "image"]
        self.audio = [item for item in message_list if item["type"] == "audio"]
        self.video = [item for item in message_list if item["type"] == "video"]
        self.file = [item for item in message_list if item["type"] == "file"]

def webui(func: Callable) -> Callable:
    webui_listeners.append(func)
    return func

# 函数调用处理逻辑
async def handle_function_call(function_call: Dict[str, Any]) -> Dict[str, Any]:
    func_name = function_call.get("name")
    args = function_call.get("args", {})
    if func_name == "get_weather":
        city = args.get("city")
        if not city:
            return {"error": "缺少城市参数"}
        return {"weather": f"{city} 的天气：晴天，25°C"}
    elif func_name == "calculate":
        expression = args.get("expression")
        if not expression:
            return {"error": "缺少表达式参数"}
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result}
        except Exception as e:
            return {"error": f"计算错误: {str(e)}"}
    else:
        return {"error": f"未知函数: {func_name}"}

# 非流式请求
async def gemini_request(history: List[Dict[str, Any]], base_url: str = BASE_URL, apikey: str = API_KEY, model: str = MODEL) -> str:
    url = f"{base_url}/v1beta/models/{model}:generateContent?key={apikey}"
    logger.info(f"发送非流式请求到: {url}")
    payload = {
        "contents": history,
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "safetySettings": [
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 16384,
            "responseMimeType": "text/plain"
        }
    }
    if TOOLS:
        payload["tools"] = [{"function_declarations": TOOLS}]
    
    logger.debug(f"发送的 POST 数据: {json.dumps(payload, indent=2)}")
    
    headers = {"Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            logger.info(f"收到响应，状态码: {response.status_code}")
            logger.debug(f"收到的响应内容: {response.text}")
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                logger.error(f"API 返回错误: {data['error']['message']}")
                raise HTTPException(status_code=400, detail=data["error"]["message"])
            
            candidate = data["candidates"][0]["content"]
            parts = candidate.get("parts", [])
            for part in parts:
                if "functionCall" in part:
                    func_response = await handle_function_call(part["functionCall"])
                    history.append({
                        "role": "function",
                        "parts": [{"functionResponse": {
                            "name": part["functionCall"]["name"],
                            "response": func_response
                        }}]
                    })
                    return await gemini_request(history)
            content = parts[0]["text"]
            logger.info(f"非流式响应内容: {content}")
            return content
    except httpx.RequestError as e:
        logger.error(f"网络请求失败: {str(e)}")
        raise HTTPException(status_code=503, detail=f"无法连接到 API: {str(e)}")
    except httpx.HTTPStatusError as e:
        logger.error(f"API 返回状态错误: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

# 流式请求（优化分块逻辑，确保完整句子和代码块）
async def gemini_stream_request(history: List[Dict[str, Any]], base_url: str = BASE_URL, apikey: str = API_KEY, model: str = MODEL) -> AsyncGenerator[str, None]:
    url = f"{base_url}/v1beta/models/{model}:streamGenerateContent?key={apikey}"
    logger.info(f"发送流式请求到: {url}")
    payload = {
        "contents": history,
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "safetySettings": [
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 16384,
            "responseMimeType": "text/plain"
        }
    }
    if TOOLS:
        payload["tools"] = [{"function_declarations": TOOLS}]
    
    logger.debug(f"发送的 POST 数据: {json.dumps(payload, indent=2)}")
    headers = {"Content-Type": "application/json"}

    async def generate() -> AsyncGenerator[str, None]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    logger.info("开始接收流式响应")
                    full_content = ""
                    first_chunk_sent = False  # 跟踪是否已发送第一个数据块
                    text_buffer = ""
                    code_buffer = ""
                    in_code_block = False
                    json_buffer = ""

                    async for chunk in response.aiter_text():
                        logger.debug(f"原始数据块: {chunk}")
                        cleaned_chunk = chunk.strip()
                        if cleaned_chunk.startswith('[') and not json_buffer:
                            cleaned_chunk = cleaned_chunk[1:]
                        elif cleaned_chunk == ']':
                            break
                        elif cleaned_chunk.startswith(','):
                            cleaned_chunk = cleaned_chunk[1:]
                        
                        json_buffer += cleaned_chunk

                        try:
                            data = json.loads(json_buffer)
                            json_buffer = ""

                            if "candidates" in data and data["candidates"]:
                                parts = data["candidates"][0]["content"].get("parts", [])
                                for part in parts:
                                    if "text" in part:
                                        content = part["text"]
                                        logger.info(f"流式数据块内容: {content}")

                                        lines = (text_buffer + content).splitlines(keepends=True)
                                        text_buffer = ""
                                        if content and not content.endswith('\n'):
                                            text_buffer = lines[-1]
                                            lines = lines[:-1]

                                        for line in lines:
                                            if in_code_block:
                                                if "```" in line:
                                                    code_buffer += line.split("```")[0]
                                                    full_content += f"```{code_buffer}```\n"
                                                    yield f"data: {json.dumps({'content': f'```{code_buffer}```', 'start_stream': not first_chunk_sent, 'end_stream': False})}\n\n"
                                                    logger.info(f"发送流式数据块 - 内容: ```{code_buffer}```, start_stream: {not first_chunk_sent}, end_stream: False")
                                                    first_chunk_sent = True
                                                    await asyncio.sleep(0.2)
                                                    in_code_block = False
                                                    code_buffer = ""
                                                    remaining = line.split("```", 1)[1]
                                                    if remaining:
                                                        full_content += remaining
                                                        yield f"data: {json.dumps({'content': remaining, 'start_stream': not first_chunk_sent, 'end_stream': False})}\n\n"
                                                        logger.info(f"发送流式数据块 - 内容: {remaining}, start_stream: {not first_chunk_sent}, end_stream: False")
                                                        first_chunk_sent = True
                                                        await asyncio.sleep(0.1)
                                                else:
                                                    code_buffer += line
                                            else:
                                                if "```" in line:
                                                    pre_content, _, code_start = line.partition("```")
                                                    if pre_content:
                                                        full_content += pre_content
                                                        yield f"data: {json.dumps({'content': pre_content, 'start_stream': not first_chunk_sent, 'end_stream': False})}\n\n"
                                                        logger.info(f"发送流式数据块 - 内容: {pre_content}, start_stream: {not first_chunk_sent}, end_stream: False")
                                                        first_chunk_sent = True
                                                        await asyncio.sleep(0.1)
                                                    in_code_block = True
                                                    code_buffer = code_start
                                                else:
                                                    full_content += line
                                                    yield f"data: {json.dumps({'content': line, 'start_stream': not first_chunk_sent, 'end_stream': False})}\n\n"
                                                    logger.info(f"发送流式数据块 - 内容: {line}, start_stream: {not first_chunk_sent}, end_stream: False")
                                                    first_chunk_sent = True
                                                    await asyncio.sleep(0.1)

                        except json.JSONDecodeError:
                            logger.debug("JSON 未完成，继续缓冲")
                            continue

                    if text_buffer:
                        full_content += text_buffer
                        yield f"data: {json.dumps({'content': text_buffer, 'start_stream': not first_chunk_sent, 'end_stream': False})}\n\n"
                        logger.info(f"发送流式数据块 - 内容: {text_buffer}, start_stream: {not first_chunk_sent}, end_stream: False")
                        first_chunk_sent = True
                        await asyncio.sleep(0.1)
                    if in_code_block and code_buffer:
                        full_content += f"```{code_buffer}```"
                        yield f"data: {json.dumps({'content': f'```{code_buffer}```', 'start_stream': not first_chunk_sent, 'end_stream': False})}\n\n"
                        logger.info(f"发送流式数据块 - 内容: ```{code_buffer}```, start_stream: {not first_chunk_sent}, end_stream: False")
                        first_chunk_sent = True
                        await asyncio.sleep(0.2)

                    logger.info(f"流式响应完成，总内容: {full_content}")
                    if full_content:
                        conversation_history.setdefault("default_user", []).append({
                            "role": "model",
                            "parts": [{"text": full_content}]
                        })
                        yield f"data: {json.dumps({'content': '', 'start_stream': False, 'end_stream': True})}\n\n"
                        logger.info(f"发送流式结束 - 内容: '', start_stream: False, end_stream: True")
        except httpx.RequestError as e:
            logger.error(f"请求发送错误: {e}")
            yield f"data: {json.dumps({'error': f'请求发送错误: {e}', 'start_stream': False, 'end_stream': True})}\n\n"
        except httpx.HTTPStatusError as e:
            logger.error(f"流式请求失败，状态码: {e.response.status_code}, 错误: {e.response.text}")
            yield f"data: {json.dumps({'error': f'HTTP错误: {e.response.status_code} - {e.response.text}', 'start_stream': False, 'end_stream': True})}\n\n"
        except Exception as e:
            logger.error(f"发生未知错误: {e}")
            yield f"data: {json.dumps({'error': f'未知错误: {e}', 'start_stream': False, 'end_stream': True})}\n\n"

    return generate()

# 构造提示元素
async def gemini_prompt_elements_construct(message_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info(f"构造提示元素，输入消息列表: {json.dumps(message_list, indent=2)}")
    prompt_elements = []

    for item in message_list:
        if item["type"] == "text":
            prompt_elements.append({"text": item["content"]})
        elif item["type"] == "image":
            img = Image(base64=item["source"]["base64"] if "base64" in item["source"] else None,
                        byte=item["source"]["byte"] if "byte" in item["source"] else None,
                        mime_type=item["source"].get("mime_type"))
            prompt_elements.append(await img.to_dict())  # 只返回 inline 数据
        elif item["type"] == "audio":
            audio = Audio(base64=item["source"]["base64"] if "base64" in item["source"] else None,
                          byte=item["source"]["byte"] if "byte" in item["source"] else None,
                          mime_type=item["source"].get("mime_type"))
            base64_data = (await audio.to_dict())["inline_data"]["data"]
            if len(base64.b64decode(base64_data)) > 20 * 1024 * 1024:  # 示例阈值
                file_uri = await upload_to_gemini_media(base64.b64decode(base64_data), audio.source["mime_type"])
                prompt_elements.append({"fileData": {"mimeType": audio.source["mime_type"], "fileUri": file_uri}})
            else:
                prompt_elements.append(await audio.to_dict())
        elif item["type"] == "video":
            video = Video(base64=item["source"]["base64"] if "base64" in item["source"] else None,
                          byte=item["source"]["byte"] if "byte" in item["source"] else None,
                          mime_type=item["source"].get("mime_type"))
            base64_data = (await video.to_dict())["inline_data"]["data"]
            if len(base64.b64decode(base64_data)) > 20 * 1024 * 1024:
                file_uri = await upload_to_gemini_media(base64.b64decode(base64_data), video.source["mime_type"])
                prompt_elements.append({"fileData": {"mimeType": video.source["mime_type"], "fileUri": file_uri}})
            else:
                prompt_elements.append(await video.to_dict())
        elif item["type"] == "file":
            file = CustomFile(base64=item["source"]["base64"] if "base64" in item["source"] else None,
                              byte=item["source"]["byte"] if "byte" in item["source"] else None,
                              mime_type=item["source"].get("mime_type"))
            base64_data = (await file.to_dict())["inline_data"]["data"]
            if len(base64.b64decode(base64_data)) > 20 * 1024 * 1024:
                file_uri = await upload_to_gemini_media(base64.b64decode(base64_data), file.source["mime_type"])
                prompt_elements.append({"fileData": {"mimeType": file.source["mime_type"], "fileUri": file_uri}})
            else:
                prompt_elements.append(await file.to_dict())

    logger.info(f"生成的提示元素: {json.dumps(prompt_elements, indent=2)}")
    return prompt_elements

# WebSocket 端点
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    clients[client_id] = websocket
    logger.info(f"客户端 {client_id} 已连接")
    
    # 非流式发送连接成功消息
    await send_message(client_id, [Text("已连接到服务器")])

    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"从客户端 {client_id} 接收到消息: {data}")
            message_data = json.loads(data)
            await handle_message(client_id, message_data)
    except Exception as e:
        logger.error(f"WebSocket 错误 (客户端 {client_id}): {str(e)}")
        # 非流式发送错误消息
        await send_message(client_id, [Text(f"WebSocket 错误: {str(e)}")])
    finally:
        del clients[client_id]
        logger.info(f"客户端 {client_id} 已断开")

# 处理 WebSocket 消息
async def handle_message(client_id: str, message_data: Dict[str, Any]):
    message_list = message_data.get("message", [])
    is_streaming = message_data.get("isStreaming", False)

    if not message_list:
        logger.warning(f"客户端 {client_id}: 消息列表为空，跳过处理")
        await send_message(client_id, [Text("消息为空，无法处理")])  # 默认非流式
        return

    event = WebUIEvent(message_list, client_id)
    for listener in webui_listeners:
        try:
            logger.debug(f"客户端 {client_id}: 调用监听函数: {listener.__name__}")
            await listener(event)
        except Exception as e:
            logger.error(f"客户端 {client_id}: 监听函数 {listener.__name__} 执行错误: {str(e)}")

    if len(message_list) == 1 and message_list[0].get("type") == "text" and message_list[0].get("content") == "/clear":
        logger.info(f"客户端 {client_id}: 接收到清除命令，清除对话历史")
        conversation_history["default_user"] = []
        await send_message(client_id, [Text("聊天记录已清除")])  # 默认非流式
        return
    
    if message_list[0].get("type") == "text" and message_list[0].get("content").startswith("/"):
        logger.info(f"客户端 {client_id}: 跳过 AI 对话")
        return

    user_id = "default_user"
    if user_id not in conversation_history:
        logger.info(f"客户端 {client_id}: 初始化新对话历史，用户: {user_id}")
        conversation_history[user_id] = []

    current_prompt = await gemini_prompt_elements_construct(message_list)
    history = conversation_history[user_id]
    history.append({"role": "user", "parts": current_prompt})

    if is_streaming:
        logger.info(f"客户端 {client_id}: 流式模式，处理用户: {user_id}")
        stream_generator = await gemini_stream_request(history)
        await send_message(client_id, stream_generator, is_streaming=True)  # 流式发送
    else:
        logger.info(f"客户端 {client_id}: 开始非流式处理，用户: {user_id}")
        try:
            answer = await gemini_request(history)
            logger.debug(f"客户端 {client_id}: 非流式响应: {answer}")
            history.append({"role": "model", "parts": [{"text": answer}]})
            conversation_history[user_id] = history
            await send_message(client_id, [Text(answer)])  # 默认非流式
        except Exception as e:
            logger.error(f"客户端 {client_id}: 非流式处理错误: {str(e)}")
            await send_message(client_id, [Text(f"处理错误: {str(e)}")])  # 默认非流式

# 发送消息到 WebSocket 客户端
async def send_message(client_id: str, message_list: List[Any], is_streaming: bool = False):
    if client_id not in clients:
        logger.warning(f"客户端 {client_id}: 不存在，跳过发送")
        return

    if not is_streaming:
        combined_messages = []
        text_content = ""

        for msg in message_list:
            if hasattr(msg, 'to_dict'):
                if asyncio.iscoroutinefunction(msg.to_dict):
                    msg_dict = await msg.to_dict()  # 不传 upload 参数，默认 inline
                else:
                    msg_dict = msg.to_dict()
                if "text" in msg_dict:
                    text_content += msg_dict["text"]
                elif "inline_data" in msg_dict:
                    mime_type = msg_dict["inline_data"]["mime_type"]
                    filename = getattr(msg, "source", {}).get("filename", "unknown_file")
                    if mime_type.startswith("image/"):
                        combined_messages.append({
                            "type": "image",
                            "source": {
                                "base64": msg_dict["inline_data"]["data"],
                                "mime_type": mime_type,
                                "filename": filename
                            }
                        })
                    elif mime_type.startswith("audio/"):
                        combined_messages.append({
                            "type": "audio",
                            "source": {
                                "base64": msg_dict["inline_data"]["data"],
                                "mime_type": mime_type,
                                "filename": filename
                            }
                        })
                    elif mime_type.startswith("video/"):
                        combined_messages.append({
                            "type": "video",
                            "source": {
                                "base64": msg_dict["inline_data"]["data"],
                                "mime_type": mime_type,
                                "filename": filename
                            }
                        })
                    else:
                        combined_messages.append({
                            "type": "file",
                            "source": {
                                "base64": msg_dict["inline_data"]["data"],
                                "mime_type": mime_type,
                                "filename": filename
                            }
                        })
                elif "type" in msg_dict and msg_dict["type"] == "node":
                    combined_messages.append(msg_dict)  # 直接添加 Node 的 JSON 结构
                elif "fileData" in msg_dict:
                    filename = getattr(msg, "source", {}).get("filename", "uploaded_file")
                    combined_messages.append({
                        "type": "file",
                        "source": {
                            "fileUri": msg_dict["fileData"]["fileUri"],
                            "mime_type": msg_dict["fileData"]["mimeType"],
                            "filename": filename
                        }
                    })
            else:
                combined_messages.append(msg)

        if text_content:
            combined_messages.insert(0, {"type": "text", "content": text_content})

        if combined_messages:
            message_json = json.dumps(combined_messages)
            await clients[client_id].send_text(message_json)
            logger.info(f"客户端 {client_id}: 非流式消息已发送: {message_json}")
    else:
        if not isinstance(message_list, AsyncGenerator):
            logger.error(f"客户端 {client_id}: 流式模式下 message_list 必须是 AsyncGenerator")
            return
        async for chunk in message_list:
            await clients[client_id].send_text(chunk)
            logger.info(f"客户端 {client_id}: 流式数据块已发送: {chunk}")
            await asyncio.sleep(0.1)

# WebUI 事件监听器
@webui
async def on_message(event: WebUIEvent):
    logger.info(f"收到 WebUI 消息: {json.dumps(event.__dict__, indent=2)}")
    if "/test" in event.plain:
        image = Image(path='C:/Users/spawner/Downloads/《Break the Cocoon》封面.jpg')
        video = Video(path='C:/Users/spawner/Downloads/QQ2025224-232140.mp4')
        audio = Audio(path='C:/Users/spawner/Downloads/Break the Cocoon_3.mp3')
        custom_file = CustomFile(path='C:/Users/spawner/Downloads/部署文档.pdf')
        await send_message(event.client_id, [
            Text("你好"),
            Text("！11111"),
            image,
            video,
            audio,
            custom_file
        ])

# 主函数
# main 函数
def main():
    parser = argparse.ArgumentParser(description="启动 AI 聊天后端服务器")
    parser.add_argument("--port", type=int, default=8000, help="服务器运行的端口号")
    args = parser.parse_args()

    port = args.port
    if not (1 <= port <= 65535):
        logger.error(f"无效的端口号: {port}，使用默认端口 8000")
        port = 8000

    # 挂载静态文件目录
    app.mount("/", StaticFiles(directory=os.path.dirname(__file__), html=True), name="static")

    frontend_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(frontend_path):
        frontend_url = f"http://127.0.0.1:{port}/index.html"
        logger.info(f"服务器启动成功，请访问前端页面: {frontend_url}")
        print(f"请在浏览器中打开: {frontend_url}")
    else:
        logger.warning("未找到 index.html 文件，请确保前端文件与 main.py 在同一目录")

    uvicorn.run(app, host="127.0.0.1", port=port)

if __name__ == "__main__":
    main()
