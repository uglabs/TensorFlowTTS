(1) ipynb
https://github.com/TensorSpeech/TensorFlowTTS/tree/master/examples/mfa_extraction
(2) bash ttsexamples/mfa_extraction/scripts/prepare_mfa.sh
(3) python ttsexamples/mfa_extraction/run_mfa.py --corpus_directory ./libritts --output_directory ./mfa/parsed --jobs 8
(4) python ttsexamples/mfa_extraction/txt_grid_parser.py \
  --yaml_path ttsexamples/fastspeech2_libritts/conf/fastspeech2libritts.yaml \
  --dataset_path ./libritts \
  --text_grid_path ./mfa/parsed \
  --output_durations_path ./libritts/durations \
  --sample_rate 24000 

(5) tensorflow-tts-preprocess --rootdir ./libritts \
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