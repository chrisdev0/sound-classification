# Urban sound sound classification

This is a continuation of many different sources of Urban sound classification implementations

## Usage
1. Put the UrbanSound8K folder in the same folder as the scripts (or create a link to the folder)
2. Create the following folders
   - **models**
   - **summary**
   - **upload**
   - **processed_audio**
3. Run
``pip install -r requirements.txt``
   - ``pip install tensorflow-gpu`` if you want to use your graphics card
      - note that additional software might be required for tensorflow-gpu to work
4. Run the scripts in the following order
   1. ``feature_extraction.py``
   2. ``feature_merge.py``
   3. ``model_creator.py``
5. The model is now created and is ready to be used for classifying sounds
6. Follow the instructions in ``urban_sound_classifier.py`` to start the flask server
7. Classify sound files by using ``sound_poster.py <path to soundfile.wav>`` 