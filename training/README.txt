=== PRETRAINED MODELS ===

OpenL3: OpenL3 is an open-source deep learning model that can be used for music and audio analysis. It provides music embeddings that capture high-level information about the music, such as timbre and rhythm. These embeddings can be used to detect changes in music or segment music into different sections based on the similarity or dissimilarity of the embeddings.

MusicVAE: MusicVAE is a generative model developed by Google's Magenta project. It is trained on a large dataset of MIDI music and can learn to generate new music compositions. However, it can also be used for music analysis tasks such as segmenting music. By analyzing the latent space representations learned by MusicVAE, you can detect meaningful changes and segment music accordingly.

MAD (Music Auto-Tagging): MAD is a pre-trained deep learning model for music analysis developed by OpenAI. Although it is primarily designed for music tagging, it can also be utilized for segmenting music. By extracting features from different time segments of the music and feeding them into the model, you can identify segment boundaries where the music undergoes significant changes.

MuNet: MuNet is a pre-trained deep learning model developed for music analysis by the Lakh MIDI Dataset project. It is trained on a large dataset of MIDI files and can predict chord progressions and musical structures. By using MuNet, you can segment music based on harmonic changes and structural patterns.

VGGish: VGGish is a pre-trained deep learning model developed by Google that can extract audio embeddings from music and other audio signals. It was originally designed for audio classification tasks but can also be used for change detection. By comparing embeddings across different segments of the music, you can identify significant changes.

YAMNet: YAMNet is another pre-trained audio classification model developed by Google. It can recognize and classify a wide range of environmental sounds, including various musical genres. YAMNet can be used to detect changes in music by analyzing the predicted classes or embeddings across different segments of the audio.

MAGNet: MAGNet (Music and Audio in Generative Adversarial Networks) is a pre-trained generative model for music analysis developed by OpenAI's Magenta project. It can be used for tasks such as chord estimation, melody generation, and music generation. By analyzing the generated music or learned latent space, you can potentially identify changes in the music.