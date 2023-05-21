import contextlib
import wave
from abc import ABCMeta, abstractmethod
import os




class AbstractStaticConversationInput():
        def __init__(self, filetype, fpath, selected_source_lang = "ja", num_speakers = 0):
                self.__filetype = filetype
                self.__file_path = fpath
                self.__selected_source_lang = selected_source_lang
                self.__num_speakers = num_speakers
        
        @property
        def selected_source_lang(self):
                return self.__selected_source_lang
        @selected_source_lang.setter
        def selected_source_lang(self, selected_source_lang):
                self.__selected_source_lang = selected_source_lang
        @property
        def num_speakers(self):
                return self.__num_speakers
        @num_speakers.setter
        def num_speakers(self, num_speakers):
                self.__num_speakers = num_speakers
        @property
        def file_path(self):
                return self.__file_path
                
        @property
        def filetype(self):
                return self.__filetype
        
        def reset(self):
                self.__file_path = ""
                self.__filetype = ""
                self.__num_speakers = 0
                self.__selected_source_lang = "ja"
        
        def __str__(self):
                return f"filetype: {self.__filetype}, file_path: {self.__file_path}, selected_source_lang: {self.__selected_source_lang}, num_speakers: {self.__num_speakers}"
        
        def __repr__(self):
                return f"filetype: {self.__filetype}, file_path: {self.__file_path}, selected_source_lang: {self.__selected_source_lang}, num_speakers: {self.__num_speakers}"

        @abstractmethod
        def size(self):
                pass
        
        @abstractmethod
        def duration(self):#A：どういう意味？Q：音声の長さを返す
                pass
        
        
class WavStaticConversationInput(AbstractStaticConversationInput):
        def __init__(self, filetype, fpath ,  selected_source_lang = "ja", num_speakers = 0):#デフォデフォルトオーギュメントは後ろに
                self.__filetype = filetype
                if self.__filetype != "wav":
                        raise Exception("filetype must be wav")
                self.__file_path = fpath
                self.__selected_source_lang = selected_source_lang
                self.__num_speakers = num_speakers
                #サイズ計算
                self.__size =os.path.getsize(self.__file_path)
                with contextlib.closing(wave.open(self.__file_path, 'r')) as f:
                        self.__frames = f.getnframes()
                        self.__rate = f.getframerate()
                        self.__duration = self.__frames / float(self.__rate)
        
        @property
        def frames(self):
                return self.__frames
        @property
        def rate(self):
                return self.__rate
        @property
        def duration(self):
                return self.__duration
        @property
        def size(self):
                return self.__size
        
        def __str__(self):
                return f"filetype: {self.__filetype}, file_path: {self.__file_path}, selected_source_lang: {self.__selected_source_lang}, num_speakers: {self.__num_speakers}, frames: {self.__frames}, rate: {self.__rate}, duration: {self.__duration}, size: {self.__size}"
        
        def __repr__(self):
                return f"filetype: {self.__filetype}, file_path: {self.__file_path}, selected_source_lang: {self.__selected_source_lang}, num_speakers: {self.__num_speakers}, frames: {self.__frames}, rate: {self.__rate}, duration: {self.__duration}, size: {self.__size}"


class AbstractLine():
        def __init__(self, lang, character_code, text, speaker, start_time, end_time):
                self.__lang = lang
                self.__character_code = character_code
                self.__text = text
                self.__speaker = speaker
                self.__start_time = start_time
                self.__end_time = end_time
                self.__duration = end_time - start_time
                self.__length = len(text)

        @property
        def lang(self):
                return self.__lang
        @property
        def character_code(self):
                return self.__character_code
        @property
        def text(self):
                return self.__text
        @property
        def speaker(self):
                return self.__speaker
        @property
        def start_time(self):
                return self.__start_time
        @property
        def end_time(self):
                return self.__end_time
        @property
        def duration(self):
                return self.__duration
        @property
        def length(self):
                return self.__length

class JapaneseLine(AbstractLine):
        def __init__(self, lang, character_code, text, speaker, start_time, end_time):
                self.__lang = lang
                if self.__lang != "ja":
                        raise Exception("lang must be ja")
                self.__character_code = character_code
                self.__text = text
                self.__speaker = speaker
                self.__start_time = start_time
                self.__end_time = end_time
                self.__duration = end_time - start_time
                self.__length = len(text)

class LinesBuffer():
        def __init__(self):
                self.__lines = []
                self.__open = True

        @property
        def num_lines(self):
                return self.__num_lines
        @property
        def lines(self):
                return self.__lines
        

        def add_line(self, line):
                if self.__open == False:
                        return False
                else:
                        self.open = False
                        self.__lines.append(line)
                        self.__open = True
                        return True
        
        def get_lines(self):
                if self.__open == False:
                        return False
                return self.__lines
        
        def pop_lines(self):
                if self.__open == False:
                        return False
                self.__open = False
                lines = self.__lines
                self.__lines = []
                self.__open = True
                return lines
        
        def reset(self):
                self.__lines = []
                self.__open = False
                

        














with gr.Tab("Whisper Transcribe Japanese Audio"):
        gr.Markdown(f'''
              <div>
              <h1 style='text-align: center'>Whisper Transcribe Japanese Audio</h1>
              </div>
              Transcribe long-form microphone or audio inputs with the click of a button! The fine-tuned
              checkpoint <a href='https://huggingface.co/{MODEL_NAME}' target='_blank'><b>{MODEL_NAME}</b></a> to transcribe audio files of arbitrary length.
          ''')
        microphone = gr.inputs.Audio(source="microphone", type="filepath", optional=True)
        upload = gr.inputs.Audio(source="upload", type="filepath", optional=True)
        transcribe_btn = gr.Button("Transcribe Audio")
        text_output = gr.Textbox()
        with gr.Row():         
            gr.Markdown('''
                ### You can test by following examples:
                ''')
        examples = gr.Examples(examples=
              [ "sample1.wav", 
                "sample2.wav", 
                ],
              label="Examples", inputs=[upload])
        transcribe_btn.click(transcribe, [microphone, upload], outputs=text_output)