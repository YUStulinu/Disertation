# Analysis and Generation of Visual Scenes with Deep Learning

A master's dissertation project with two modules joined by one theme: a system that **understands** a scene and one that **builds** a new, realistic scene.

- **Module 1 – Scene analysis:** real-time recognition of the actions of a person filmed with a webcam (walking, sitting down, falling, lifting an object, standing still), from the body skeleton.
- **Module 2 – Scene generation:** automatic cut-out of an object from a photo and its realistic placement in another background, with colour harmonization and a shadow. The use case demonstrated: museum objects placed in other halls.
- **Modules 1 + 2 together:** in front of the camera, whoever holds out their hands gets a museum object in them, at its real size and harmonized with the camera image, in real time.

The detailed results of both modules (figures, tables and frames from the recording of the live demo) are in `results.html`, also online: **[yustulinu.github.io/Disertation/results.html](https://yustulinu.github.io/Disertation/results.html)**.

## Project structure

```text
python_research/         research, training and evaluation code (Python)
  common/                code shared by all scripts (paths, skeleton, hands, segmentation, composition,
                         harmonization, quality metrics, plots, the objects-in-hands scene)
  data/raw/              the recorded skeleton clips, one folder per action
  data/benchmark_photos/ the real photos used by the harmonization benchmark
  data/objects/          the object library of the interactive demo: catalog.json
                         (the photo and real size of every object) + the RGBA cut-outs
  data/*_results/        the generated figures and CSV tables
  models/                the trained LSTM action classifier (.pt / .onnx) and the MediaPipe Pose (lite)
                         and Hand Landmarker models; YOLOv8, SAM 2.1 and Real-ESRGAN download here on first use
  third_party/           pretrained harmonization networks (inference code and weights only):
                         Harmonizer (ECCV 2022, CC BY-NC-SA 4.0) and PCT-Net (CVPR 2023, MPL-2.0)
cpp_app/                 the live Qt 6 / C++ app: action recognition (ONNX Runtime) and the harmonized
                         museum objects in hands
photos/                  the photos of Module 2: vases, weapons, armour, helmets, shields and other objects
                         from The Met Open Access collection (CC0), museum halls and a town square from
                         Wikimedia Commons; CREDITS.md lists the author and licence of every image
images/                  app screenshots: python_app/ (the Python demos; python_app/demo/: frames from the
                         recording of the objects-in-hands demo, the gallery in results.html) and
                         qt_cpp_app/ (the Qt/C++ app); the S key in the apps saves straight into these folders
results.html             the results of both modules, with figures and tables (opens in a browser)
```

## Setup

```bat
cd python_research
python -m venv venv
venv\Scripts\activate
python -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

The project was developed with Python 3.14 on Windows. The first install command brings PyTorch for an NVIDIA GPU (here an RTX 2060), which speeds up segmentation, inpainting and harmonization tens of times. Without an NVIDIA GPU, skip it: everything then runs on the CPU, more slowly.

Optional, for the `intrinsic` and `reshade` methods (Intrinsic Harmonization, academic-use licence):

```bat
python -m pip install "setuptools<81" wheel
python -m pip install --no-build-isolation https://github.com/compphoto/IntrinsicCompositing/archive/main.zip
```

The network weights (YOLOv8, SAM 2.1, LaMa, Depth Anything V2, Real-ESRGAN, IC-Light, Intrinsic Harmonization) download automatically on first use. On the first run of the `intrinsic` method, PyTorch Hub may ask whether to trust the `rwightman/gen-efficientnet-pytorch` repository; answer `y`. All scripts find their files by themselves, so they can be run from any folder. VS Code is configured (`.vscode/settings.json`) to use the `python_research/venv` environment.

## Module 1 – Action recognition

### How it works

1. **Skeleton extraction:** MediaPipe Pose detects 33 body landmarks in every camera frame.
2. **Time window:** 30 samples of the skeleton are taken from the last ~1.9 seconds, at equal time intervals (15 per second), by interpolating between frames. The window is defined in time, not in frames, so the model sees a movement the same way whatever the speed of the camera or of the computer.
3. **Normalization:** the points are centred on the hip midpoint and scaled by the shoulder–hip distance, so an action looks the same wherever the person stands and however far they are from the camera.
4. **Classification:** the network receives the positions of the points and computes, inside itself, the **velocity** of every point as well (falling and sitting down differ mostly in speed). An LSTM layer runs through the 30 samples, its outputs are averaged over time, and a final layer predicts the action.
5. **Live display:** the probabilities are smoothed over ~0.2 s so that the label does not flicker. Below 60% confidence, "uncertain" is shown instead of a guessed action. The "idle" class (standing still, including with the hands held out, as when holding an object) makes the moments without any action recognized as such, instead of being forced into an action or shown as "uncertain".
6. **The final app:** the model is exported to ONNX and run in the C++/Qt app with ONNX Runtime, without PyTorch.

### Workflow

| Step                                                                 | Command                                                                                                                                                                                      |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Recording the clips (5 s each, 10 per action)                        | `python record_session.py` or a single action: `--action idle` (`--save-video` also keeps the video)                                                                                         |
| Dataset summary / replaying a clip                                   | `python playback_data.py`, `python playback_data.py data/raw/walking/1.npy`                                                                                                                  |
| Training the model used live (on all clips)                          | `python train.py`                                                                                                                                                                            |
| Evaluation on unseen clips: metrics, CSV, charts                     | `python evaluate.py [--include-robustness]`                                                                                                                                                  |
| ONNX export (verified against PyTorch)                               | `python export_onnx.py`                                                                                                                                                                      |
| The live demo, identical to the C++ app                              | `python verify_pipeline.py` (skeleton only: `--pose-only`; S saves the image to `images/python_app/<action>_<n>.png`)                                                                        |
| The live robustness test, with timed trials                          | `python robustness_test.py --condition normal` (keys 1–5 = the action, U = delete the last trial), then `--report`                                                                           |
| Old vs new model on held-out clips (the 4 initial actions)           | `python train.py --holdout 2 --out models/action_classifier_holdout.pt`, then `python evaluate.py --compare models/action_classifier_v1.pt --checkpoint models/action_classifier_holdout.pt` |

### Evaluation protocol

The data are split by **whole clips**, never by windows, so the windows of one recording never end up both in training and in testing. No score is computed on training clips. The reported result is that of **5-fold cross-validation**: 5 models are trained, and every clip is tested once, by a model that never saw it.

The architecture was also chosen by cross-validation, among several variants (LSTM, bidirectional GRU, temporal convolutional network, with and without velocity). Adding the velocity brought the largest gain.

| Metric (5-fold cross-validation) | Initial model, 4 actions | Final model, 4 actions | Final model, 5 actions (with idle) |
| -------------------------------- | ------------------------ | ---------------------- | ---------------------------------- |
| Window accuracy                  | 78.4%                    | 84.2%                  | **88.5%**                          |
| Clip accuracy (majority vote)    | 80.0%                    | 87.5%                  | **94.0%**                          |
| Macro F1                         | 0.77                     | 0.83                   | **0.88**                           |
| MCC                              | 0.71                     | 0.80                   | **0.86**                           |

The dataset has 50 clips of 5 seconds, 10 per action. Adding the "idle" class also improved the other classes: the model is no longer forced to fit the still moments into an action. "Idle" has an F1 of 0.93. The hardest class remains "sitting" (recall 70%, vs 53% without "idle"), most often confused with "falling" (16% of its windows) and "lifting_object" (9%): all three end with the person down or bent over.

On the live path (time-sampled window, smoothing and the 60% threshold), run on the clips that each cross-validation fold's model did not see, 47 of the 50 clips are recognized correctly; while the person stands still, the label shown is "idle" 96% of the time and "uncertain" only 2%.

### Live robustness test

The final model was tested live, with the camera, in 5 conditions, with timed trials: the action is chosen, then come 3 s to get into position and 4 s during which what the program shows is recorded (the same path as the C++ app: time-sampled window, smoothing, the 60% threshold). There was one trial per action in every condition (25 trials, ~600 frames per condition).

| Condition (live test)              | Frame accuracy | Correct trials | Falling | Idle | Lifting | Sitting | Walking |
| ---------------------------------- | -------------- | -------------- | ------- | ---- | ------- | ------- | ------- |
| normal                             | 99.8%          | 5/5            | 100%    | 99%  | 100%    | 100%    | 100%    |
| fast movements (fast)              | 95.8%          | 5/5            | 100%    | 82%  | 100%    | 97%     | 100%    |
| low light (low_light)              | 96.8%          | 5/5            | 90%     | 94%  | 100%    | 100%    | 100%    |
| side view (side_angle)             | 92.2%          | 5/5            | 68%     | 96%  | 97%     | 100%    | 100%    |
| far from the camera (far_distance) | 95.3%          | 5/5            | 93%     | 83%  | 100%    | 100%    | 100%    |

All 25 trials are recognized correctly (by the action shown in most of their frames). Per frame, the hardest condition is the side view, especially falling (68%): seen from the side, falling looks like sitting down. "Idle" drops to 82–83% with fast movements and far from the camera, where it is sometimes confused with walking. With one trial per action and condition, the per-action figures are indicative; the per-condition ones rest on ~600 frames each.

### The C++ app

The Qt 6 / C++ app in `cpp_app` runs both parts live: action recognition and the museum objects in hands (described below, in [Modules 1 + 2 together](#in-the-c-app)). It builds with CMake (it needs Qt 6, OpenCV and ONNX Runtime); for demos, use the **Release** configuration (in Debug, OpenCV runs unoptimized and the image has fewer frames per second):

```bat
cmake -S cpp_app -B build -DCMAKE_PREFIX_PATH=C:/Qt/6.11.1/msvc2022_64 -DORT_DIR=C:/onnxruntime/onnxruntime-win-x64-1.27.0
cmake --build build --config Release
build\Release\ActionRecognitionApp.exe
```

The `bin` folders of Qt and OpenCV must be on the `PATH`; `onnxruntime.dll` is copied next to the executable automatically. In VS Code, the same settings are in `.vscode/settings.json`, and the configuration is chosen from the CMake bar. The app:

- finds the project folder by itself and starts `python_research/pose_server.py` with the Python of the `venv`; another interpreter can be chosen with the `ACTION_APP_PYTHON` environment variable;
- reads the camera in Full HD (1920×1080), in a separate thread, and sends the Python server a 960×540 copy of each frame (the skeleton and hand models work on small images anyway), at most two frames at a time: while the server works on one, the next one is already waiting, and the delay does not build up;
- opens its window centred on the screen, at 1.3 × the size of the camera image or smaller, so that a Full HD image fits;
- samples the window in time, smooths the probabilities and applies the confidence threshold exactly as in Python;
- shows the action, its confidence and the skeleton detection rate (fps) over the camera image, and prints every change of action to the console.

Arguments: `--no-objects` starts only the action recognition (the skeleton computed on every frame), `--video clip.mp4` runs on a video, in a loop, instead of the camera. The **S** key saves the image shown, with everything drawn over it, to `images/qt_cpp_app/`: `objects_<n>.png`, or with `--no-objects` `<action>_<n>.png` (for example `idle_1.png`); the numbers continue, nothing is overwritten.

## Module 2 – Image composition and harmonization

### How it works

For a moved object to look as if it belongs in the picture, the pipeline deals in turn with every cue by which the eye recognizes a composite: the shape of the cut-out, its edges, colour and light, sharpness, noise and the shadow.

1. **Segmentation:** YOLOv8-seg detects the object, and **SAM 2.1** (Segment Anything 2) computes its precise mask from the detected box. The mask follows thin parts (handles, fingers) and leaves out what YOLO's coarse mask grabs around the object (for example a vase's glass stand).
2. **Inpainting:** the hole left in the source image is filled, classically (TELEA) or with the LaMa network.
3. **Matted cut-out:** the edge of the mask becomes a soft transition that follows the photo's real contours (guided filter, He et al. 2010). The colours of the edge pixels are then cleaned of the old background (decontamination): the compositing equation I = αF + (1−α)B is solved for F, with B estimated from the inpainted image. This removes the light or dark outline (the "halo") around pasted objects.
4. **Composition in the scene's geometry:** the **Depth Anything V2** network estimates the distance in metres of every background pixel. Knowing the object's real height (`--height`, for example 0.51 m for an amphora), its size in the image follows from the camera formula h = f · H / Z (focal length times real height, divided by distance). A plane fitted to the floor's 3D points also gives the floor's perspective, used for the shadow. If the object has to be enlarged, **Real-ESRGAN** first restores its detail (super-resolution), so that it does not come out blurry.
5. **Colour and light harmonization:** the mean of three networks trained on iHarmony4 (Harmonizer, PCT-Net ViT, PCT-Net CNN). Each of them errs in different cases, and the mean compensates their errors. Optionally, the object can also be **relit** by the scene's light (`--harmony intrinsic`, `reshade` or `iclight`, see the table below).
6. **"Camera" match:** the object's sharpness is brought to the scene's (blurring or sharpening, with a strength found automatically by measuring the blur), and the sensor noise it lacks is added.
7. **Reflection:** on glossy floors (marble, lacquered parquet), the object is mirrored, blurred and fading (`--reflection 0–1`).
8. **Shadow:** a contact shadow under the base, drawn in the floor's perspective (default), or a cast shadow.

All the new steps can be switched off (`--no-sam`, `--no-matting`, `--no-sr`, `--harmony harmonizer`, `--no-camera`, `--reflection 0`, `--shadow none`), which allows a direct comparison with the previous version of the pipeline.

### Workflow

| Step                                                                                 | Command                                                                                                                       |
| ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| Segmentation and cut-out of the object (RGBA cut-out + mask)                         | `python segment.py --image photo.jpg --class person`                                                                          |
| YOLOv8 nano vs medium (+ IoU/Dice with `--gt-mask`)                                  | `python segment.py --image photo.jpg --class vase --model n m`                                                                |
| Classical and LaMa inpainting (`--eval` = score against the ground truth)            | `python inpaint.py --image photo.jpg --mask mask.png --methods telea ns lama --eval`                                          |
| Composition (also saves the mask of the placed object, `_mask.png`)                  | `python compose.py --fg object.png --bg background.jpg`                                                                       |
| Harmonization with every method + comparison figure (`--ablation` = the VGG depth)   | `python harmonize.py --composed X.jpg` or `python harmonize.py --all --ablation`                                              |
| The whole pipeline / the museum objects demo                                         | `python pipeline.py ...`, `python museum_demo.py --source amphora.jpg --background hall.jpg --height 0.51 --pos 0.5 0.86`     |
| Shadows                                                                              | `python add_shadow.py --batch data/museum_demo [--mode cast --angle 20]`                                                      |
| Metrics (benchmark with a ground truth)                                              | `python metrics.py`                                                                                                           |

### Harmonization methods

| Method               | Description                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ensemble` (default) | **The mean of the three networks below.** Their errors partly cancel out, so the ensemble is better than each network on its own.                                                                                                                                                                                                                                                             |
| `harmonizer`         | Trained network (Harmonizer, Ke et al., ECCV 2022): an EfficientNet-B0 trained on iHarmony4 predicts six image filters (temperature, brightness, contrast, saturation, highlights, shadows).                                                                                                                                                                                                  |
| `pctnet`             | Trained network (PCT-Net, Guerreiro et al., CVPR 2023) with a ViT backbone: predicts a separate colour transform for every pixel.                                                                                                                                                                                                                                                             |
| `pctnet_cnn`         | The same PCT-Net method with a convolutional backbone.                                                                                                                                                                                                                                                                                                                                         |
| `vgg`                | Baseline without training: the object's colour transform is optimized so that the statistics of its VGG16 features approach those of the background (layers relu1_2 … relu4_3).                                                                                                                                                                                                             |
| `reinhard`           | Classical colour transfer in Lab space (Reinhard et al., 2001).                                                                                                                                                                                                                                                                                                                                |
| `histogram`          | Histogram matching on every colour channel.                                                                                                                                                                                                                                                                                                                                                    |
| `intrinsic`          | **Intrinsic harmonization** (Careaga et al., SIGGRAPH Asia 2023): the image is decomposed into albedo (the surface's own colour) and shading; the object's albedo is harmonized and the object is re-shaded under the light estimated from the background (spherical harmonics computed from the scene's shading and normals). It also changes the direction of the light on the object, not only its colour. |
| `reshade`            | The colours of `ensemble`, then only the re-shading step of `intrinsic`. The albedo step of the original method over-corrects objects with strong colours (terracotta turns olive), so here the colour stays the ensemble's.                                                                                                                                                                 |
| `iclight`            | **IC-Light** (Zhang et al., ICLR 2025): a diffusion model (Stable Diffusion 1.5) that regenerates the object lit as in the background. Since diffusion can alter details (the painting of a vase), only the change of light is kept, as a smooth gain map applied to the image at full resolution.                                                                                        |

Every method changes **only the object**, using its mask at its position in the composite. On an RTX 2060 (6 GB), `intrinsic`/`reshade` and `iclight` take 12–20 s per image (peak memory 4.7 GB and 4.4 GB), the other methods under a second; the two heavy models do not fit on the GPU at the same time, so each one frees the other.

### Evaluation protocol

Harmonization is evaluated with the protocol of the iHarmony4 benchmark: **real photos** are taken, the object is recoloured at random, and each method has to bring it back to its natural look. The original photo is the ground truth, so the metrics are meaningful: MSE/PSNR on the whole image, fMSE/fPSNR on the object only, and fSSIM.

| Method (30 cases)                            | fMSE ↓  | fPSNR ↑     | fSSIM ↑   |
| -------------------------------------------- | ------- | ----------- | --------- |
| No harmonization                             | 626     | 21.3 dB     | 0.876     |
| Reinhard (classical)                         | 2080    | 16.1 dB     | 0.785     |
| Histogram matching (classical)               | 2649    | 15.0 dB     | 0.714     |
| VGG16 statistics (no training)               | 543     | 22.3 dB     | 0.908     |
| Harmonizer (previous version)                | 280     | 24.8 dB     | 0.932     |
| PCT-Net ViT                                  | 275     | 24.6 dB     | 0.930     |
| PCT-Net CNN                                  | 282     | 24.5 dB     | 0.929     |
| **Ensemble: Harmonizer + PCT-Net ViT + CNN** | **191** | **26.5 dB** | **0.949** |
| Intrinsic Harmonization (relighting)         | 1001    | 19.9 dB     | 0.836     |
| Ensemble + re-shading                        | 1195    | 20.2 dB     | 0.869     |
| IC-Light (relighting)                        | 743     | 21.2 dB     | 0.866     |

The mean of the three networks is the best method on every metric: it reduces the error on the object by 32% compared with Harmonizer (the previous version of the pipeline) and by 69% compared with the unharmonized image. The classical methods make the result worse, because they assume the object should have the colours of the background, which is usually not true (a red vase in a beige hall). The training-free VGG method improves the result slightly, but is unstable.

The relighting methods (`intrinsic`, `reshade`, `iclight`) score lower on this benchmark, as expected: the object is only recoloured in its own photo, so its lighting is already right, and any change of shading is penalized. The benchmark measures colour correction, not lighting. On real composites, relighting can help visually (IC-Light on a person), but it can also exaggerate (terracotta vases become too saturated, and the albedo of the `intrinsic` method turns greenish). That is why `ensemble` remains the default and relighting is optional.

Inpainting is evaluated by moving the object's mask to an object-free area of the same image, where the real pixels are known. LaMa is the most accurate on both images: on the photo with the person it reaches 20.1 dB PSNR, against 15.2 dB (TELEA) and 15.6 dB (Navier-Stokes), and on the museum showcase 11.7 dB and an SSIM of 0.30, against 0.18–0.19 for the classical methods.

### Pretrained models used

| Model                                    | Role                                    | Licence                           |
| ---------------------------------------- | --------------------------------------- | --------------------------------- |
| YOLOv8-seg (Ultralytics)                 | detection and the coarse mask           | AGPL-3.0                          |
| SAM 2.1 (Meta)                           | the precise mask                        | Apache-2.0                        |
| LaMa                                     | inpainting                              | Apache-2.0                        |
| Depth Anything V2 Metric Small           | depth in metres                         | Apache-2.0                        |
| Real-ESRGAN x4plus                       | super-resolution                        | BSD-3-Clause                      |
| Harmonizer, PCT-Net                      | colour harmonization                    | CC BY-NC-SA 4.0, MPL-2.0          |
| Intrinsic Harmonization                  | relighting (`intrinsic`, `reshade`)     | academic use only                 |
| IC-Light + Realistic Vision 5.1 (SD 1.5) | diffusion relighting (`iclight`)        | Apache-2.0, CreativeML OpenRAIL-M |
| MediaPipe Pose + Hand Landmarker         | the body and hand skeletons             | Apache-2.0                        |

## Modules 1 + 2 together – museum objects in your hands

Whoever holds their hands out towards the camera gets a museum object in them. Every person gets their own objects: the hands of different people never hold the same object.

| A person's held-out hands         | The object                                                                                               |
| --------------------------------- | -------------------------------------------------------------------------------------------------------- |
| one hand                          | a small object (up to 25 cm)                                                                             |
| both hands, less than 50 cm apart | one large object, held with both hands (first the one that best fits the distance between the hands)     |
| both hands, more than 70 cm apart | one small object in each hand                                                                            |

With one hand, the object and the harmonization are changed manually (the N and H keys). With both hands up, the hands are busy, so – whether they hold one large object or two small ones – the objects change by themselves every 4 seconds, in shuffled order, as if dealt from a deck of cards: every object appears once before any comes again (then the deck is shuffled again), never the same one in both hands, and never passed from one hand to the other at the next change; the N key draws from the same deck. Each object is shown 2 s harmonized and 2 s raw, in the same rhythm for both hands. The rhythm has a clock per person, started when the person raises the second hand: neither a hand the detector loses for a moment nor a switch between one large and two small objects resets it. A raised hand stays counted as raised (even with half-bent fingers) until it hangs down or is missing for more than 0.8 s. The difference made by the harmonization depends on the scene's light: in a room with white, neutral light, like the objects' studio photos, the change is small (a few percent of brightness and colour); under warm or coloured light (a lamp, a yellow bulb), it is much more visible.

Between 50 and 70 cm nothing changes (hysteresis). A switch between one large and two small objects happens only after the new situation has held for 0.35 s and at least 1.5 s after the previous switch, so that the objects can be seen; the transition is a ~0.1 s cross-fade.

The object is held the way the hands are held:

| The hands                                         | The object                                                                                                                                                                                                                                                        |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| open (the palm towards the camera or up)          | it stands on the palm; with two hands as far apart as it is wide, it is held at its belly, from the sides; a sword lies across both palms                                                                                                                         |
| closed as in a grip (the back of the hand shows)  | it is held by its handles: the handles go into the hands, and the fingers stay in front of them; a weapon held in one hand is held by its grip; an object without handles by its widest upper part (the rim of a helmet or of a shield), and the bust and the swords as by open hands |

**The object library** has 30 objects from The Met Open Access collection (CC0), 12 for one hand and 18 for both, each at its real size (the figure `data/objects/catalog_preview.jpg` shows them at the same scale):

| For                                                  | The objects                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| one hand (up to 25 cm, or weapons held by the grip)  | the hippopotamus "William" (Egyptian faience, 11 cm), three small amphorae (16–21 cm), a lion-shaped aquamanile (~1200, 21 cm), Filippo Negroli's parade helmet (Milan, 1543, 24 cm), an armour gauntlet with gilded bands (Valentin Siebenbürger, ~1535, 29 cm), a flintlock pistol (~1700–1730, 32 cm), a bronze spearhead (1200–800 BCE, 40 cm), a Japanese javelin in its sheath (44 cm), a Mughal dagger with a jade horse-head hilt, in its sheath (49 cm) and a Saxon war hammer (57 cm)                                                                         |
| both hands (over 25 cm, or the swords lying flat)    | a celestial globe with clockwork carried by Pegasus (Gerhard Emmoser, 1579, 27 cm), a Greek helmet of the Corinthian type (31 cm), an etched morion (32 cm), a Roman marble bust (42 cm), five large amphorae (46–78 cm), a painted Indian shield (dhal, 52 cm), two horse-head defences (Dutch, ~1620, 54 cm, and Japanese, with horns, 56 cm), a breastplate by Giovan Paolo Negroli (59 cm), the shield of King Henry II of France (64 cm), a Neapolitan presentation sabre with a gilded scabbard (94 cm), a sword with an Ottoman hilt, in its scabbard (102 cm), an infantry armour from Nuremberg (~1600, 125 cm) and a two-handed sword (~1400–1450, 146 cm) |

The cut-out also works for objects YOLO does not know (helmets, the breastplate, the bust, the weapons): SAM starts from the largest weak detection in the photo, and where YOLO sees nothing or only a part (the celestial globe: the sphere, without Pegasus and the base; the two-handed sword, the horse-head defences, the armour), the catalog gives the box of the whole object. Small holes that SAM leaves inside an object are filled before the matting, and the soft edge of a cut-out is at most as wide as that of a 1600 px photo: on the 4000 px photos of the swords, the blades would otherwise come out partly see-through. The museum photographs its weapons upright. The swords are turned to lie flat in their cut-outs (the photos stay unchanged), have their real length instead of a height in the catalog and lie across both palms at their full length; the dagger, the spears and the war hammer are held in one hand, standing on the open palm by their lower end or gripped by the handle in a fist. Objects without handles are held in a grip by their widest upper part (the rim of a helmet or of a shield, the ring of the globe), and the bust and the swords as by open hands.

### How it works

1. **The hands:** MediaPipe Hand Landmarker finds 21 points per hand. The detector on the whole frame tracks two hands; the hands of other people are searched for in an enlarged crop around every wrist the skeleton model (Pose, for several people) finds, which also finds the small hands of people farther away.
2. **"Held-out hand":** the hand does not hang by the body and is either open (at least three straight fingers, measured as 3D angles, with the palm not turned to the floor, as with a hand resting on a knee or on a table) or closed as in a grip (at least three bent fingers).
3. **People and pairs:** the skeleton model finds the people and keeps their identities from one frame to the next: every person takes over the nearest one seen in the last second, so someone the model loses for a moment stays the same person. Every hand is tied to the wrist of the nearest arm, hence to a particular person. A hand whose body is not found in a frame stays with the person it belonged to (a hand does not change owner, so it cannot end up paired with someone else's hand), and if the person still gets a new identity or their arms are swapped, their objects and their rhythm follow them. A hand lost for a few frames keeps its place for 0.8 s, so that the pair does not break. Two hands with no body in view that the detector gives the same side (left or right, common with palms held up) are told apart by their position. The distance between a person's two hands, in metres, decides whether they hold one object together (the rules above).
4. **Real size:** the model also gives the 3D coordinates of the hand in metres, so the distance between the knuckles, in pixels and in metres, gives the scale of the image (pixels per metre) at the hand. The scale is taken from the hands, not from the body: hands held out towards the camera are much closer to it than the shoulders. The object is drawn at its real height, from `catalog.json`; held with two hands, it is enlarged or shrunk by at most 30% so that the hands touch its belly or its handles.
5. **No jitter:** the position, the size and the distance between the hands go through a One Euro filter (Casiez et al., CHI 2012), which smooths strongly while the hands are still and follows them without visible lag when they move. A switch between "open" and "grip" happens only after several frames in a row, so that the object does not jump.
6. **Real-time harmonization:** the harmonization of Module 2 runs in a separate process, twice a second. For every object, a 3×4 colour matrix that turns the raw colours into the harmonized ones is computed; the object is re-coloured once per update, not once per frame. From the networks' result, the change of brightness and contrast (the scene's light) is taken in full, but only 25% of the change of colour, so that the vases keep their real colours (the reason and the measurements are [below](#the-scenes-light-the-vases-real-colours)). Each update runs one of the three networks of the ensemble in turn, and the matrix is the mean of the last three. Since fitting the matrix is linear in the target image, the result is practically the ensemble's, at a third of the cost. The object keeps the sharpness of its photo (it is not blurred to match the hands, see below); its grain is the camera's noise (next step).
7. **Full HD:** the camera is read at its maximum resolution, 1920×1080. The hands and the skeletons are found in a 960×540 copy (the models work on small images anyway), and the objects are drawn into the Full HD image, from cut-outs at the full resolution of the museum's photos (mostly 900–2300 px on the long side; the swords, lying flat, ~3600 px long, from the museum's 4000 px originals), hence with the detail of their paintings and engravings. The camera noise is measured on the Full HD image itself: cameras smooth and compress the image, so their noise does not grow when the image is enlarged (the camera used has σ ≈ 0.25 grey levels at 1080p and 0.42 on a 480-row copy).
8. **Realism:** the hands stay in front of an object held at its belly or by its handles; a contact shadow appears under an object standing on the palm.
9. **Module 1 in parallel:** the action classifier runs on the same stream and shows what the person is doing.

### Workflow

| Step                                                                     | Command                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The object library (cut out once with YOLOv8m + SAM 2.1 + matting)       | `python build_catalog.py` (`--force` redoes them all; cut-outs at the photos' resolution: up to 1600 px; the large objects and the long weapons held in one hand up to 2400 px; the swords 4000 px, since on the palms they reach more than 3000 px in width)                                  |
| The live demo                                                            | `python ar_objects.py` (keys: H = harmonization on/off (the objects stay, unharmonized, for comparison), O = objects shown/hidden, N = another object, D = the hand landmarks, S = save the image to `images/python_app/objects_<n>.png`, Q or Esc = quit; closing the window quits too)      |
| Another rhythm for the objects held with both hands                      | `python ar_objects.py --cycle 6 --alternate 3` (default: 4 s per object and 2 s harmonized / raw; `--cycle 0` keeps the best-fitting object, `--alternate 0` keeps it always harmonized)                                                                                                     |
| The live demo, recorded                                                  | `python ar_objects.py --out demo.mp4` (H.264 through x264 / imageio-ffmpeg, ~55–80 MB per minute in Full HD depending on the camera image's noise; plays in browsers too; live, at real speed)                                                                                              |
| Another camera resolution                                                | `python ar_objects.py --resolution 1280x720` (default `1920x1080`, Full HD; the camera picks the closest resolution it has)                                                                                                                                                                  |
| How much of the harmonization's colour change the vases get              | `python ar_objects.py --colour 0` (brightness only; `1` = the networks' whole change; default 0.25)                                                                                                                                                                                         |
| Measuring this choice (benchmark + the vases in real scenes, figure)     | `python live_harmony_eval.py`                                                                                                                                                                                                                                                                 |
| The detail kept by harmonized vases (before / now, figure)               | `python live_harmony_detail.py`                                                                                                                                                                                                                                                               |
| On a video or on photos                                                  | `python ar_objects.py --video clip.mp4 --out result.mp4`, `python ar_objects.py --image photo.jpg`                                                                                                                                                                                           |

A new object is added with its photo in `photos/` and a line in `data/objects/catalog.json` (the photo and the real height in metres; optionally `"class"`, the YOLO class to look for, `"vase"` by default), then `python build_catalog.py`. An object YOLO does not recognize with confidence (a statue, a helmet, a mask) is cut out by SAM from the largest weak detection in the photo. For an object YOLO finds only part of, `"box"` gives its box in the photo (fractions of the width and height: `[x1, y1, x2, y2]`), from which SAM cuts it out; `"handles": false` marks an object without handles (the bust, the swords), held in a grip as by open hands. For weapons: `"width_m"` gives the real length of an object lying flat (a sword) instead of its height, `"hands"`: 1 or 2 says with how many hands it is held, whatever its height, `"handles"`: a number gives the row of its grip (a fraction of the height, from the top), and `"rotate"`: 90 / 180 / 270 turns the cut-out (the photo stays unchanged).

On the laptop with an RTX 2060, the live demo runs at ~25–30 frames per second in Full HD (the camera gives 30), also with a large vase held in both hands close to the camera. The work is split over threads that run in parallel: one reads the camera, another finds the nearby hands in the 960×540 copy of the newest frame (18–30 ms), and the main thread draws the objects into the Full HD image (~2–12 ms; a large vase taller than the frame, ~7–12 ms), the text and the recording. The limit is now finding the hands (~25–30 ms per frame); before, a large vase brought the image down to 8–10 frames per second, because drawing it took 62 ms (90% of the frames under 189 ms): the cut-out was resized in floating point for every new size, and the hands always move. Now the cut-out is prepared once, with its colours extended under the transparent area (pull-push), so that resizing does not bring dark edges in; an object much smaller than its photo is resized whole with INTER_AREA, as before, and kept while its size and the harmonization do not change, while one close to the size of its photo or larger (over ~70%, a large vase held close) is sampled bilinearly every frame, only where it is seen (at these ratios, practically the same as INTER_AREA); the colour is applied once per update, on the cut-out, and the compositing is done directly in the 8-bit image (`cv2.blendLinear`), in reused arrays. The vases look the same as before (0.1–0.5 grey levels of mean difference, a median PSNR of 48–52 dB per vase), without the dark ring on their edges. The status text is redrawn only when it changes, and the recording passes the frame to ffmpeg without a copy. The skeletons of all people, the search for distant hands and the action classifier run in another thread, 15 times a second; the harmonization runs in its own process, with a reduced priority and few CPU threads, and its networks run on the GPU. The networks are loaded ~6 s after the start; until then the objects appear without harmonization. The cut-outs of the 30 objects are prepared at the start, in parallel (~1 s in Python, ~0.5 s in C++), so that the first appearance of an object does not delay the image. The window opens centred on the screen, at 1.3 × the size of the camera image or smaller, so that a Full HD image fits, and it can be resized.

### The scene's light, the vases' real colours

The harmonization networks are trained (on iHarmony4) to repair objects whose colours are **wrong**, so they pull the object's colours towards the scene's. The museum photos, however, have the **right** colours; compared with the camera's scene, only the light differs. Applied in full, the networks changed the vases' own colours a lot: in a sunny square they washed them out (the orange of an amphora became a greyish yellow), and in a warm hall they turned them yellow. That is why, live, the change of brightness and contrast is taken from the networks in full, but only part of the change of colour (`--colour`). `python live_harmony_eval.py` measures this choice with exactly the live code: on the harmonization benchmark (30 cases from real photos, the ground truth being the original photo) and on the 8 vases put into 5 real scenes (4 museum halls and a square):

| Share of the colour change taken | fMSE ↓ | fPSNR ↑ | fSSIM ↑ | Vase saturation (mean, min–max) | Vases within ±15% of their real saturation |
| -------------------------------- | ------ | ------- | ------- | ------------------------------- | ------------------------------------------ |
| no harmonization                 | 626    | 21.3 dB | 0.876   | 1.00                            | 100%                                       |
| 100% (previous version)          | 186    | 26.8 dB | 0.952   | 0.84 (0.29–1.60)                | 30%                                        |
| 50%                              | 201    | 26.5 dB | 0.948   | 0.92 (0.61–1.28)                | 58%                                        |
| **25% (default)**                | 219    | 26.2 dB | 0.945   | **0.95 (0.80–1.14)**            | **90%**                                    |
| 0% (brightness only)             | 246    | 25.7 dB | 0.940   | 0.99 (0.95–1.05)                | 100%                                       |

The change of brightness and contrast alone brings 81% of the harmonization's gain on the benchmark (from 21.3 to 25.7 dB, out of 26.8 dB). With 25% of the colour change, the vases practically keep the colours of their museum photos, and in a warm light they still get a warm tint. On the benchmark, only 0.6 dB is lost, because there the objects have wrong colours by construction. The live version (the matrix fitted on the networks' result) is even slightly better on the benchmark than the networks applied pixel by pixel (26.8 dB vs 26.5 dB): the global matrix smooths out their local artefacts. The figure `python_research/data/metrics_results/live_harmony_colour.jpg` shows the vases as photographed, with the whole colour change and with 25%.

### The painting and inscriptions of harmonized vases

Once harmonized, a vase used to lose almost all its detail: the painting, the incised lines, the inscriptions. The "camera" match blurred it down to the sharpness measured on the hands holding it, but the skin of the hands is smooth and always reads as blurred, so the blur sat at its cap (1.5 px at 480 rows, i.e. 3.4 px in Full HD) in 55% of the updates, with the median right at the cap. The camera, however, is not blurry: measured on the sharpest edges of its own frames, its transitions take 1.25 px (10–90%), less than those of a vase drawn without any blur (1.75 px). So a vase held in the hands, at the same distance as they are, is no longer blurred: the harmonization changes only its light and tone (the colour matrix, with 25% of the colour change) and adds the camera's grain. In addition, the cut-outs now keep the full resolution of the photos (973–1314 px, not cropped to 1080 px), and the two large amphorae, which the hands bring closest to the camera (up to ~1800 px on screen), are made from the museum's originals downscaled to 2400 px (2031 and 2099 px): on screen they are downscaled, not enlarged.

| Vase                    | On screen | Detail kept before | Detail kept now | SSIM before | SSIM now |
| ----------------------- | --------- | ------------------ | --------------- | ----------- | -------- |
| Amphora Small Geometric | 500 px    | 6%                 | 100%            | 0.51        | 0.999    |
| Amphora Small Cypriot   | 500 px    | 9%                 | 100%            | 0.61        | 0.999    |
| Amphora Small Etruscan  | 500 px    | 5%                 | 100%            | 0.59        | 0.999    |
| Amphora Red Figure      | 1000 px   | 5%                 | 113%            | 0.60        | 0.995    |
| Jar Terracotta          | 1000 px   | 6%                 | 114%            | 0.53        | 0.994    |
| Amphora Black Figure    | 1000 px   | 5%                 | 111%            | 0.59        | 0.996    |
| Amphora Large Attic     | 1500 px   | 7%                 | 110%            | 0.70        | 0.996    |
| Amphora Large Apulian   | 1500 px   | 7%                 | 115%            | 0.69        | 0.995    |

Measured with `live_harmony_detail.py`: every vase at the size it has in the Full HD demo (one hand ~500 px, two hands ~1000 px, a large amphora held close ~1500 px), in the light of a warm museum hall (the matrix the live harmonization fits there), compared with the museum photo in the same light: the fine detail kept (the energy of the Laplacian, 100% = the photo's) and the structural similarity (SSIM) of the brightness. Over 100% means that the bilinear sampling of large vases seen up close is a touch sharper than the reference filter. Figure: `data/metrics_results/live_harmony_detail.jpg`.

### In the C++ app

The Qt/C++ app (`build\Release\ActionRecognitionApp.exe`, see [The C++ app](#the-c-app) for building it) shows the objects exactly like the Python demo: the same rules, the same automatic rhythm with both hands up, the same harmonization and the same keys (H, O, N, D, S, Q or Esc; closing the window quits too; S saves the image to `images/qt_cpp_app/objects_<n>.png`). The action classifier runs in parallel and shows the action at the top, while the objects and the state of the harmonization are listed at the bottom. The work is split as follows:

- **Python (`pose_server.py`)** does the perception and the decisions, with exactly the code of the demo: the hands, the people, the scene (which object, where, how it is held, the harmonized / raw phase) and the harmonization, in its own process. For every frame it sends back, as JSON, the description of the objects (`HandheldScene.describe`): the cut-out, its size and position in the frame, how it is held, its opacity, the 3×4 colour matrix, the sharpness and noise to match, and the points of the hands that have to stay in front of the object.
- **C++ (`object_renderer.cpp`)** composes the image step by step like `HandheldScene.render`: it starts from the cut-out with its colours extended under the transparent area (no edges of another colour), resizes it whole with INTER_AREA when it is much smaller than its photo (kept while its size and the harmonization do not change) or samples bilinearly only the visible part when it is close to the photo's size or larger (over ~70%, a large vase held close), applies the colour matrix once per harmonization update and the sharpness at the drawn size, composes directly in the 8-bit image (`cv::blendLinear`), in reused buffers, adds fresh sensor noise every frame, draws the contact shadow under objects standing on the palm and puts the hands back in front of the objects held at their belly or by their handles.

The C++ compositing was checked pixel by pixel against the Python one, on the same scene, with all three ways of holding, harmonized and raw (including the raw phase of the automatic rhythm), with blurring and with sharpening, partial opacity and an object cut by the edge of the frame, both at 640×480 and on the Full HD path (the scene computed in the small copy, the objects drawn into a 1080-row image): the images are identical for raw objects and for a large vase close to the camera (1569 px, taller than the frame) and differ by at most 2 levels out of 255 otherwise (4 pixels out of 48,000), from rounding, and the added noise has the requested deviation (3.98 measured, for 4). Drawing three objects takes ~0.9 ms at 640×480 and ~4 ms in Full HD (Release; before, 2.5 and 10 ms), and a large vase moving close to the camera ~8.5 ms: the grain comes from a noise texture (a new random window of it every frame), the mask of the hands is computed only around the hand, and the status text is redrawn only when it changes.

The objects are drawn on the very frame in which the hands were found, so they do not lag behind them. The camera is read in Full HD, in a separate thread (opening it takes a few seconds, and the window does not wait for it); the Python server receives a 960×540 copy of the frame and answers with positions and sizes at the Full HD scale. The server has two frames at a time: while it finds the hands in one, the next one is already read and decoded in a separate thread, and C++ prepares the frame after it while it draws the answer. With the camera, the image has ~27–30 frames per second in Release, also with the objects in hand (the camera gives 30; the limit is finding the hands, ~25–30 ms per frame; in Debug, fewer); after many minutes of continuous running, when the laptop's processor heats up and lowers its frequency, it can drop towards ~20. On exit (Q, Esc, closing the window), the Python server and the harmonization process stop too; if the app is killed (End Task), they stop by themselves within a second.

## Credits and licences

- The photos in `photos/`: The Met Open Access (CC0) and Wikimedia Commons (CC BY / CC BY-SA / public domain); `photos/CREDITS.md` lists the author and licence of every image.
- `python_research/third_party/`: Harmonizer (Ke et al., ECCV 2022; CC BY-NC-SA 4.0, non-commercial) and PCT-Net (Guerreiro et al., CVPR 2023; MPL-2.0), with their original licence notices.
- The pretrained models and their licences are listed in [Pretrained models used](#pretrained-models-used).
