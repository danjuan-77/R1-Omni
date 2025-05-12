import os
import argparse
import tempfile
import traceback

import json
from humanomni import model_init, mm_infer
from humanomni.utils import disable_torch_init
from transformers import BertTokenizer
from tqdm import tqdm
from typing import List, Optional
from moviepy.editor import (
    AudioFileClip,
    concatenate_audioclips,
    ImageClip,
    concatenate_videoclips,
    ColorClip,
    VideoFileClip,
)
tempfile.tempdir = "/home/tuwenming/Projects/HAVIB/tmp"

# 设置环境变量
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
pmp_avl_ans_format = "answer={'category1_id1': '[x_min, y_min, x_max, y_max]', 'category2_id2': '[x_min, y_min, x_max, y_max]'}"
avl_cls_list = ['dog', 'clarinet', 'banjo', 'cat', 'guzheng', 'tree', 'lion', 'tuba', 
        'ukulele', 'flute', 'piano', 'person', 'violin', 'airplane', 'bass', 'pipa', 
        'trumpet', 'accordion', 'saxophone', 'car', 'lawn-mower', 'cello', 'bassoon', 
        'horse', 'guitar', 'erhu', 'not sure', 'no available option']
prompt_avl = f"""
        ctaegories list: {avl_cls_list}
        (1) There may be multiple sounding instances, you can choose instance categories from the given categories list.
        (2) The naming format for instances is: category_id. Instance IDs start from 1, e.g., male_1, dog_2, dog_3, cat_4. 
        (3) The bbox format is: [x_min, y_min, x_max, y_max], where x_min, y_min represent the coordinates of the top-left corner. 
        (4) The bbox values should be normalized into the range of 0 and 1, e.g., [0.1, 0.12, 0.26, 0.14].
        Do not explain, you must strictly follow the format: {pmp_avl_ans_format}
    """

prompt_avlg = """
        Please output the answer in a format that strictly matches the following example, do not explain:
        answer={'frame_0': [x0_min, y0_min, x0_max, y0_max], 'frame_1': None, ..., 'frame_9': [x9, y9, w9, h9]}
        Note, 
        (1) x_min, y_min represent the coordinates of the top-left corner, while x_max, y_max for the bottom_right corner.
        (2) The bbox values should be normalized into the range of 0 and 1, e.g., [0.1, 0.12, 0.26, 0.14]. 
        (3) Frames should be ranged from frame_0 to frame_9.
    """

avqa_cls_list = ['ukulele', 'cello', 'clarinet', 'violin', 'bassoon', 'accordion', 'banjo', 'tuba', 'flute', 'electric_bass', 'bagpipe', 
        'drum', 'congas', 'suona', 'xylophone', 'saxophone', 'guzheng', 'trumpet', 'erhu', 'piano', 'acoustic_guitar', 'pipa', 'not sure', 'no available option']

havib_constants = {
    'L3_AVH': {
        'prompt_avh': "Please answer the question based on the given video.",
        'avh_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_VAH': {
        'prompt_vah': "Please answer the question based on the given audio.",
        'vah_options_list': ['yes', 'no', 'not sure'],
    },

    'L3_AVL': {
        'prompt_avl': prompt_avl,
        'avl_cls_list': avl_cls_list,
    },


    'L4_AVC': {

    },

    'L4_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L4_AVQA': {
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },

    'L5_AVLG': {
        'prompt_avlg': prompt_avlg,
    },

    'L5_AVQA': {
        'avqa_cls_list': avqa_cls_list,
        'avqa_options_list_is': ['yes', 'no', 'not sure'],
    },
}
def concat_audio(audio_paths: List[str]) -> str:
    """
    Concatenate multiple audio files into one WAV file.
    Returns the path to the temp WAV file.
    """
    clips = [AudioFileClip(p) for p in audio_paths]
    final = concatenate_audioclips(clips)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_path = tmp.name
    final.write_audiofile(out_path, fps=16000, logger=None)
    return out_path

def audio_to_video(audio_path: str, fps: int = 1, resolution=(640, 480)) -> str:
    """
    将音频转换成黑场视频，并附带原音。
    - audio_path: 输入的 wav 路径
    - fps: 帧率
    - resolution: 视频分辨率
    返回临时 mp4 文件路径。
    """
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    # 生成黑场
    clip = ColorClip(size=resolution, color=(0, 0, 0)).set_duration(duration)
    clip = clip.set_audio(audio_clip)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    clip.write_videofile(tmp.name, fps=fps, codec="libx264", logger=None)
    return tmp.name

def images_to_video(image_paths: List[str], duration: float, fps: int = 1) -> str:
    """
    Turn a list of images into a silent video of total `duration` seconds.
    Each image is shown for `duration / len(image_paths)` seconds.
    Returns the path to the temp MP4 file.
    """
    single_dur = duration / len(image_paths)
    clips = [ImageClip(p).set_duration(single_dur) for p in image_paths]
    video = concatenate_videoclips(clips, method="compose")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    video.write_videofile(out_path, fps=fps, codec="libx264", audio=False, logger=None)
    return out_path

def images_and_audio_to_video(image_paths: List[str], audio_paths: List[str], fps: int = 1) -> str:
    """
    Concatenate audio_paths into one audio, then build a video from image_paths
    that matches the audio duration, and merge them.
    Returns the path to the temp MP4 file.
    """
    # 1) build the concatenated audio
    audio_path = concat_audio(audio_paths)
    audio_clip = AudioFileClip(audio_path)
    # 2) build video from images matching audio duration
    duration = audio_clip.duration
    vid_path = images_to_video(image_paths, duration, fps=fps)
    # 3) attach audio to video
    video_clip = AudioFileClip(audio_path)  # re-open to avoid MoviePy caching issues
    base_vid = VideoFileClip(vid_path)
    final = base_vid.set_audio(audio_clip)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    final.write_videofile(out_path, fps=fps, codec="libx264", logger=None)
    return out_path 
    
def get_real_path(task_path: str, src_path: str) -> str:
    """传入taskpath和一些文件的path，构造文件的真实path

    Args:
        task_path (str): task path
        src_path (str): 每个文件的path

    Returns:
        str: 文件的真实path
    """
    temp_path = os.path.join(task_path, src_path)
    return os.path.normpath(temp_path)

def get_real_options_or_classes(d: dict) -> str:
    """Replace pseudo-options with real options text."""
    opts = d['input']['question'].get('options')
    if opts in havib_constants.get(d['task'], {}):
        opts = havib_constants[d['task']][opts]
    if opts:
        label = 'semantic categories' if 'cls' in opts else 'options'
        return f"Available {label} are: {opts}"
    return ''

def get_real_prompt(d: dict) -> str:
    """Replace pseudo-prompt with real prompt text."""
    prm = d['input']['question'].get('prompt')
    if prm in havib_constants.get(d['task'], {}):
        prm = havib_constants[d['task']][prm]
    return prm or ''

def get_real_input(d: dict) -> str:
    """Concatenate prompt, options, and question text into one input string."""
    prompt = get_real_prompt(d)
    options = get_real_options_or_classes(d)
    question = d['input']['question']['text'] or ''
    # 去掉多余的句点
    parts = [p for p in (prompt, options, question) if p]
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="HumanOmni Inference Script")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )

    parser.add_argument(
        "--task_path",
        type=str,
        required=True,
        help="Path to the task folder containing data.json and media files",
    )

    args = parser.parse_args()

    # 初始化BERT分词器
    bert_model = "bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model)

    # 禁用Torch初始化
    disable_torch_init()

    # 初始化模型、处理器和分词器
    model, processor, tokenizer = model_init(args.model_path)

    
    task_path = args.task_path
    task_name = f"L{task_path.rsplit('/', 1)[0][-1]}_{task_path.rsplit('/', 1)[-1]}"
    model_name = args.model_path.split('/')[-1]
    save_prediction_json = f'/home/tuwenming/Projects/HAVIB/eval/user_outputs/{model_name}/tasks/{task_name}.json'
    os.makedirs(os.path.dirname(save_prediction_json), exist_ok=True)
    print('>>> save res to:', save_prediction_json)
    
    
    data_json_path = os.path.join(task_path, "data.json")
    with open(data_json_path, "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    print(">>>Finished load raw data...")
    parsed_data = []
    for item in raw_data:
        inp = item.get('input', {})
        question = inp.get('question', {})
        entry = {
            'id': item.get('id'),
            'task': item.get('task'),
            'subtask': item.get('subtask', None),
            'text': get_real_input(item),
            'audio_list': inp.get('audio_list', None),
            'image_list': inp.get('image_list', None),
            'video': inp.get('video', None)
        }
        parsed_data.append(entry)

    print(">>>Finished parse raw data...")    
    
    predictions = []
    
    for data in tqdm(parsed_data):
        _id = data['id']
        _task = data['task']
        _subtask = data['subtask']
        text = data['text']
        audio_list = (
            [get_real_path(task_path, p) for p in data["audio_list"]]
            if data["audio_list"] else None
        )
        image_list = (
            [get_real_path(task_path, p) for p in data["image_list"]]
            if data["image_list"] else None
        )
        video = (
            get_real_path(task_path, data['video'])
            if data['video'] else None
        )
        text += " Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        print(f">>> text input=:{text}")
        try:
            if audio_list and not image_list and not video:
                # Case 1: 仅音频 -> 黑场视频
                audio_path = concat_audio(audio_list) if len(audio_list) > 1 else audio_list[0]
                video_path = audio_to_video(audio_path, fps=1)
                use_audio = True
                if model_name in ["R1-Omni-0.5B","HumanOmni-0.5B"]:
                    modal = 'video'
                else:
                    modal = 'audio'
                    

            elif image_list and not audio_list and not video:
                # Case 2: 仅图像 -> 无声视频 (时长5s)
                video_path = images_to_video(image_list, duration=5.0, fps=1)
                use_audio = False
                modal = 'video'
                

            elif video and not audio_list and not image_list:
                # Case 3: 仅视频
                video_path = video
                use_audio = False
                modal = 'video'
                

            elif video and audio_list:
                # Case 4: 视频+音频列表（实际上直接使用视频自带音频）
                audio_path = audio_list[0]
                if not os.path.exists(audio_path): # 去掉音频
                    video_path = video
                    use_audio = False
                    modal = 'video'
                elif not os.path.exists(video): # 去掉视频画面
                    audio_path = concat_audio(audio_list) if len(audio_list) > 1 else audio_list[0]
                    video_path = audio_to_video(audio_path, fps=1)
                    use_audio = True
                    if model_name in ["R1-Omni-0.5B","HumanOmni-0.5B"]:
                        modal = 'video'
                    else:
                        modal = 'audio'
                else:
                    video_path = video
                    use_audio = True
                    modal = 'video_audio'
                

            elif image_list and audio_list and not video:
                # Case 5: 图像+音频 -> 合成视频
                audio_path = audio_list[0]
                if not os.path.exists(audio_path): # 去掉音频
                    video_path = images_to_video(image_list, len(image_list), fps=1)
                    use_audio = False
                    modal = 'video'
                else:
                    video_path = images_and_audio_to_video(image_list, audio_list, fps=1)
                    use_audio = True
                    modal = 'video_audio'

            else:
                raise ValueError(f"Unsupported input combination for id={_id}")

            # 加载张量
            video_tensor = processor['video'](video_path)
            audio_tensor = processor['audio'](video_path)[0] if use_audio else None
        
            # 执行推理
            output = mm_infer(video_tensor, text, model=model, tokenizer=tokenizer, modal=modal, question=text, bert_tokeni=bert_tokenizer, do_sample=False, audio=audio_tensor)
        except Exception as e:
            # 捕获任何异常，并把完整 traceback 当作 output
            tb = traceback.format_exc()
            output = f"Error during inference:\n{tb}"
            
        pred_record = {
            "task": _task,
            "subtask": _subtask,
            "id": _id,
            "predict": output,
        }
        predictions.append(pred_record)
        print('>>> ans=:', pred_record)
        
    with open(save_prediction_json, 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()