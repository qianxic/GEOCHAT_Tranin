import dataclasses
from enum import auto, Enum
from typing import List, Tuple
from PIL import Image
from threading import Thread

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import process_images_demo, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer,TextStreamer
import torch
import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any


class SeparatorStyle(Enum):
    """不同的分隔符风格，用于定义对话历史中消息的分隔方式"""
    SINGLE = auto()  # 单一分隔符
    TWO = auto()     # 两种分隔符交替使用
    MPT = auto()     # MPT模型特定的分隔风格
    PLAIN = auto()   # 简单分隔，无角色标识
    LLAMA_2 = auto() # LLaMA-2模型特定的分隔风格


@dataclasses.dataclass
class Conversation:
    """对话类，保存所有对话历史记录，管理会话上下文"""
    system: str                 # 系统提示信息
    roles: List[str]            # 对话角色列表，通常为["USER", "ASSISTANT"]
    messages: List[List[str]]   # 消息历史，格式为[(role, message), ...]
    offset: int                 # 消息偏移量，用于确定从哪开始显示消息
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE  # 分隔符风格
    sep: str = "###"            # 主要分隔符
    sep2: str = None            # 次要分隔符（TWO风格使用）
    version: str = "Unknown"    # 模型版本

    skip_next: bool = False     # 是否跳过下一条消息

    def get_prompt(self):
        """
        根据对话历史和分隔符风格，生成完整的提示文本
        将所有对话历史组合成模型输入格式
        """
        messages = self.messages
        # 特殊处理带图像的消息
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        # 根据不同的分隔符风格处理消息格式
        if self.sep_style == SeparatorStyle.SINGLE:
            # 单一分隔符风格处理
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            # 两种分隔符交替使用的风格处理
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            # MPT模型特定格式处理
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            # LLaMA-2模型特定格式处理
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            # 简单分隔，无角色标识
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        """添加新消息到对话历史"""
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        """
        从对话历史中提取所有图像
        return_pil: 是否返回PIL图像对象，否则返回base64编码的图像
        """
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    # 根据不同的图像处理模式进行处理
                    if image_process_mode == "Pad":
                        # 将图像填充为正方形
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    
                    # 处理图像尺寸，保持适当的长宽比
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if longest_edge != max(image.size):
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                    
                    # 根据返回类型处理图像
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        """
        将对话历史转换为Gradio聊天机器人格式
        处理图像和文本消息，适配Gradio界面显示
        """
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    # 处理图像尺寸
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    
                    # 将图像转换为base64编码，用于在HTML中显示
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        """创建对话的深拷贝"""
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        """将对话转换为字典格式，方便序列化"""
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

default_conversation = conv_vicuna_v0
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,

    "mpt": conv_mpt,
}

class Chat:
    """
    聊天类，管理模型与用户之间的交互
    处理图像和文本输入，生成模型回复
    """
    def __init__(self, model, image_processor, tokenizer, device='cuda:0', stopping_criteria=None):
        """
        初始化聊天实例
        model: 模型实例
        image_processor: 图像处理器
        tokenizer: 分词器
        device: 设备 (cpu或cuda)
        stopping_criteria: 生成终止条件
        """
        self.device = device
        self.model = model
        self.vis_processor = image_processor
        self.tokenizer = tokenizer

        # 注释掉的代码是原始stopping_criteria的实现
        # if stopping_criteria is not None:
        #     self.stopping_criteria = stopping_criteria
        # else:
        #     stop_words_ids = [torch.tensor([2]).to(self.device)]
        #     self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self, text, conv):
        """
        处理用户输入的文本
        text: 用户输入文本
        conv: 对话实例
        """
        # 如果上一条消息是图像消息，则将当前文本添加到该消息中
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-9:] == '<image>\n':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            # 否则添加新消息
            conv.append_message(conv.roles[0], text)

    def answer_prepare(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                       repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=2000):
        """
        准备模型生成回复所需的参数
        conv: 对话实例
        img_list: 图像列表
        max_new_tokens: 最大生成令牌数
        其他参数为生成配置参数
        """
        # 添加空回复作为占位符
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # 处理输入标记，包括图像标记
        text_input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=self.device)

        # 设置停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, text_input_ids)
        
        # 处理超长输入
        current_max_len = text_input_ids.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = text_input_ids[:, begin_idx:]

        # 返回生成参数字典
        generation_kwargs = dict(
            input_ids=embs,
            images=img_list[0],
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            use_cache=True,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )
        return generation_kwargs

    # 注释掉的answer方法，用于非流式生成
    # def answer(self, conv, img_list, **kargs):
    #     generation_dict = self.answer_prepare(conv, img_list, **kargs)
    #     output_token = self.model_generate(**generation_dict)[0]
    #     output_text = self.model.llama_tokenizer.decode(output_token, skip_special_tokens=True)

    #     output_text = output_text.split('###')[0]  # remove the stop sign '###'
    #     output_text = output_text.split('Assistant:')[-1].strip()

    #     conv.messages[-1][1] = output_text
    #     return output_text, output_token.cpu().numpy()

    def stream_answer(self, conv, img_list, **kargs):
        """
        流式生成回复，允许逐字显示生成结果
        conv: 对话实例
        img_list: 图像列表
        kargs: 其他生成参数
        """
        generation_kwargs = self.answer_prepare(conv, img_list, **kargs)
        
        # 创建流式生成器
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs['streamer'] = streamer
        # 调用模型生成
        output = self.model_generate(kwargs=generation_kwargs)
        # 注释掉的线程方式
        # thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        # thread.start()
        return streamer

    def model_generate(self, *args, **kwargs):
        """
        调用模型生成文本
        实际执行生成过程，处理图像和文本输入
        """
        # 使用推理模式，确保不计算梯度
        with torch.inference_mode():
            output = self.model.generate(kwargs['kwargs']['input_ids'],
                                         images=kwargs['kwargs']['images'],
                                         do_sample=False,
                                         temperature=kwargs['kwargs']['temperature'],
                                         max_new_tokens=kwargs['kwargs']['max_new_tokens'],
                                         streamer=kwargs['kwargs']['streamer'],
                                         use_cache=kwargs['kwargs']['use_cache'],
                                         stopping_criteria=kwargs['kwargs']['stopping_criteria'])
            # 解码生成的标记
            outputs = self.tokenizer.decode(output[0,kwargs['kwargs']['input_ids'].shape[1]:]).strip()
        return output

    def encode_img(self, img_list):
        """
        编码图像，将图像转换为模型能处理的格式
        img_list: 图像列表
        """
        image = img_list[0]
        img_list.pop(0)
        if isinstance(image, str):  # 图像路径
            raw_image = Image.open(image).convert('RGB')
            image = process_images_demo([raw_image], self.vis_processor)
        elif isinstance(image, Image.Image):  # PIL图像
            raw_image = image
            image = process_images_demo([raw_image], self.vis_processor)
            image = image.to(device=self.device, dtype=torch.float16)
        elif isinstance(image, torch.Tensor):  # Tensor格式
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
        image = image.to(self.device)

        # 将处理后的图像添加回列表
        img_list.append(image)

    def upload_img(self, image, conv, img_list):
        """
        处理上传的图像
        image: 上传的图像
        conv: 对话实例
        img_list: 图像列表
        """
        # 添加图像标记到对话
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN+'\n')
        img_list.append(image)
        msg = "Received."

        return msg


# 注释掉的主函数
# if __name__ == "__main__":
#     print(default_conversation.get_prompt())
