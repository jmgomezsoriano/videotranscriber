from argparse import ArgumentParser
from sys import stdout
from typing import Optional, TextIO

import torch
from glob import glob
from moviepy.editor import VideoFileClip
from mysutils.tmp import removable_tmp
from tqdm.auto import tqdm


class VideTranscriberArgParser(object):
    @property
    def input(self) -> str:
        return self._args.input

    @property
    def output(self) -> Optional[str]:
        return self._args.output

    @property
    def lang(self) -> str:
        return self._args.lang

    @property
    def gpu(self) -> bool:
        return self._args.gpu

    def __init__(self) -> None:
        parser = ArgumentParser(description='A video transcriber.')
        self._set_arguments(parser)
        self._args = parser.parse_args()

    @staticmethod
    def _set_arguments(parser: ArgumentParser) -> None:
        parser.add_argument('-i', '--input', metavar='FILE', type=str, required=True, help='The input video file.')
        parser.add_argument('-o', '--output', metavar='FILE', type=str,
                            help='The text file to save the transcription. By default, the standard output is used.')
        parser.add_argument('-l', '--lang', metavar='LANG', type=str.lower, choices=['en', 'de', 'es'], default='en',
                            help='The transcription language.')
        parser.add_argument('-g', '--gpu', action='store_true', help='Activate to use GPU processors.')


class VideoTranscriber(object):
    def __init__(self, lang: str = 'en', gpu: bool = False) -> None:
        self.device = torch.device('gpu' if gpu else 'cpu')  # gpu also works, but our models are fast enough for CPU

        self.model, self.decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                         model='silero_stt',
                                                         language=lang,  # also available 'de', 'es'
                                                         device=self.device)
        (read_batch, split_into_batches, read_audio, prepare_model_input) = utils  # see function signature for details
        self.read_batch = read_batch
        self.split_into_batches = split_into_batches
        self.prepare_model_input = prepare_model_input

    def transcribe(self, video_file: str, output_stream: TextIO) -> None:
        with removable_tmp(suffix='.wav') as tmp_file:
            clip = VideoFileClip(video_file)
            clip.audio.write_audiofile(tmp_file)
            # download a single file, any format compatible with TorchAudio (soundfile backend)
            # torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav',
            #                                dst='speech_orig.wav', progress=True)
            audio_files = glob(tmp_file)
            batches = self.split_into_batches(audio_files, batch_size=10)
            input = self.prepare_model_input(self.read_batch(batches[0]), device=self.device)

            output = self.model(input)
            for example in tqdm(output, desc='Transcribing'):
                print(self.decoder(example.cpu()), file=output_stream)


def main(args: VideTranscriberArgParser):
    transcriber = VideoTranscriber(args.lang, args.gpu)

    if args.output:
        with open(args.output, 'wt') as output:
            transcriber.transcribe(args.input, output)
    else:
        transcriber.transcribe(args.input, stdout)


if __name__ == '__main__':
    main(VideTranscriberArgParser())
