(1) ipynb
https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples/mfa_extraction
(2) 
bash ttsexamples/mfa_extraction/scripts/prepare_mfa.sh
(3) 
python ttsexamples/mfa_extraction/run_mfa.py --corpus_directory ./libritts --output_directory ./mfa/parsed --jobs 8

(4) 
python ttsexamples/mfa_extraction/txt_grid_parser.py \
  --yaml_path ttsexamples/fastspeech2_libritts/conf/fastspeech2libritts.yaml \
  --dataset_path ./libritts \
  --text_grid_path ./mfa/parsed \
  --output_durations_path ./libritts/durations \
  --sample_rate 24000 

(5) 
tensorflow-tts-preprocess --rootdir ./libritts \
  --outdir ./dump_libritts \
  --config preprocess/libritts_preprocess.yaml \
  --dataset libritts

(6) tensorflow-tts-normalize --rootdir ./dump_libritts \
  --outdir ./dump_libritts \
  --config preprocess/libritts_preprocess.yaml \
  --dataset libritts

(7) python ttsexamples/mfa_extraction/fix_mismatch.py \
  --base_path ./dump_libritts \
  --trimmed_dur_path ./dataset/trimmed-durations \
  --dur_path ./libritts/durations

(8) bash ttsexamples/fastspeech2_libritts/scripts/train_libri.sh

=============================================================================


CUDA_VISIBLE_DEVICES=0 python ttsexamples/tacotron2/extract_duration.py \
  --rootdir ./dump_libritts/train/ \
  --outdir ./dump_libritts/train/durations/ \
  --checkpoint ./ttsexamples/tacotron2/exp/train.tacotron2.v1/checkpoints/model-120000.h5 \
  --use-norm 1 \
  --config ./ttsexamples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32 \
  --win-front 3 \
  --win-back 3

CUDA_VISIBLE_DEVICES=0 python ttsexamples/tacotron2/extract_duration.py \
  --rootdir ./dump_libritts/valid/ \
  --outdir ./dump_libritts/valid/durations/ \
  --checkpoint ./ttsexamples/tacotron2/exp/train.tacotron2.v1/checkpoints/model-120000.h5 \
  --use-norm 1 \
  --config ./ttsexamples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32 \
  --win-front 3 \
  --win-back 3




get the files


  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12jvEO1VqFo1ocrgY9GUHF_kVcLn3QaGW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12jvEO1VqFo1ocrgY9GUHF_kVcLn3QaGW" -O model-120000.h5 && rm -rf /tmp/cookies.txt

get the data

wget https://www.openslr.org/resources/60/train-clean-100.tar.gz

some solution
https://stackoverflow.com/questions/63199164/how-to-install-libcusolver-so-11

sudo ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusolver.so.11



----------------------------------------------------------------------------
LJSpeech

tensorflow-tts-preprocess --rootdir ./LJSpeech-1.1 \
  --outdir ./dump_ljspeech \
  --config preprocess/ljspeech_preprocess.yaml \
  --dataset ljspeech


tensorflow-tts-normalize --rootdir ./dump_ljspeech \
  --outdir ./dump_ljspeech \
  --config preprocess/ljspeech_preprocess.yaml \
  --dataset ljspeech

CUDA_VISIBLE_DEVICES=0 python ttsexamples/tacotron2/extract_duration.py \
  --rootdir ./dump_libritts/train/ \
  --outdir ./dump_libritts/train/durations/ \
  --checkpoint ./ttsexamples/tacotron2/exp/train.tacotron2.v1/checkpoints/model-120000.h5 \
  --use-norm 1 \
  --config ./ttsexamples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32 \
  --win-front 3 \
  --win-back 3

CUDA_VISIBLE_DEVICES=0 python ttsexamples/tacotron2/extract_duration.py \
  --rootdir ./dump_libritts/valid/ \
  --outdir ./dump_libritts/valid/durations/ \
  --checkpoint ./ttsexamples/tacotron2/exp/train.tacotron2.v1/checkpoints/model-120000.h5 \
  --use-norm 1 \
  --config ./ttsexamples/tacotron2/conf/tacotron2.v1.yaml \
  --batch-size 32 \
  --win-front 3 \
  --win-back 3
