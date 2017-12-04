install:
	pip3 install numpy scipy resampy tensorflow six
	curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
	curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
	mv vggish* src/utils/

run:
	python3 src/main.py

generate:
	python3 ./src/generateFeatures.py --wav_file sounds/sound.wav \
                                    --tfrecord_file generatedFeatures/sound.tfrecord \
                                    --checkpoint src/utils/vggish_model.ckpt \
                                    --pca_params src/utils/vggish_pca_params.npz
