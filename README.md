***向下滚动页面以浏览中文版***

------

# wav-from-sofa

**wav-from-sofa** is a specialized Python tool designed  to extract HRTF (Head-Related Transfer Function) parameters from SOFA  files and generate high-precision WAV convolution files.

It is specifically engineered to create **Virtual Surround** experiences on headphones by simulating a pair of high-fidelity  speakers placed in a perfect equilateral triangle (±30° azimuth, 3  meters distance) in front of the listener.

It generates a .wav file that uses **HeSuVi standard channel order [LL, RR, LR, RL]**, effectively facilitating seamless integration with EqualizerAPO. (For specific configuration procedures on importing the files into EqualizerAPO, please refer to the instructions detailed below.)

## Features

- **Standard Stereo Simulation:** Extracts HRTF data for speakers at ±30° azimuth and 0° elevation.
- **Physics-Based Rendering:** Preserves the natural Interaural Time Difference (ITD) embedded in the SOFA data without artificial delays.
- **High-Precision Resampling:** Supports exporting to 44.1kHz, 48kHz, 88.2kHz, 96kHz, 176.4kHz, and 192kHz.
- **HeSuVi Compatibility:** Generates 4-channel WAV files using the specific channel order required for robust True Stereo virtualization.
- **Auto-Config:** Automatically generates a configuration file (`.txt`) for immediate use with Equalizer APO.

## Dependencies

This tool requires Python 3 and the following libraries:

- `pysofaconventions`
- `netCDF4`
- `numpy`
- `scipy`

## Installation

1. Ensure you have Python installed.
2. Install the required libraries using pip:

bash

```bash
pip install pysofaconventions netCDF4 numpy scipy
```

## Usage

1. Run the script:

   bash

   ```bash
   python wav-from-sofa-v1.6.py
   ```

1. **Step 1:** Drag and drop your `.sofa` file into the terminal window (or paste the path) and press Enter.
2. **Step 2:** The script will validate the file and detect coordinate conventions.
3. **Step 3:** Select your desired output sample rate (e.g., 48000 Hz).
4. **Step 4:** Enter the path where you want to save the `.wav` file (e.g., `D:\HRTF\my_preset.wav`).
5. **Result:** The script will generate two files:
   - `my_preset.wav` (The convolution data)
   - `my_preset.wav.txt` (The configuration file for Equalizer APO)

## Integration with Equalizer APO (Important)

To ensure the best experience and avoid common issues (silence,  clipping, or permission errors), please follow these steps strictly:

### 1. Windows Sample Rate Matching

**Crucial:** The sample rate of the generated WAV file **must match** the sample rate of your playback device in Windows.

- Go to **Control Panel -> Sound -> Playback**.
- Right-click your device (DAC/Headphones) -> **Properties -> Advanced**.
- Ensure the "Default Format" matches the sample rate you selected in the script (e.g., 24 bit, 48000 Hz).

### 2. File Placement & Permissions

Equalizer APO often lacks permission to read files outside its own installation directory.

- **Action:** Copy **BOTH** the generated `.wav` file and the `.txt` file into the Equalizer APO config folder.
- **Default Path:** `C:\Program Files\EqualizerAPO\config\`

### 3. Editing the Configuration Path

Since you moved the files to the `config` folder, you may need to edit the generated `.txt` file.

- Open the `.txt` file with Notepad.
- Locate the line starting with `Convolution:`.
- **Change:** `Convolution: D:\Original\Path\my_preset.wav`
- **To:** `Convolution: my_preset.wav` (Relative path) **OR** `Convolution: C:\Program Files\EqualizerAPO\config\my_preset.wav` (Absolute path).

### 4. Importing into Editor

1. Open **Configuration Editor.exe**.
2. Add a **Preamp** at the very top. Set its gain to a negative value based on the "**Peak Gain**" reading in EqualizerAPO's **Analysis Panel**, ensuring that the "Peak Gain" value **no longer appears in red**. This prevents clipping and distortion, as convolution processing typically increases overall signal gain.
3. Add an **Include** command (Green Plus button -> Control -> Include).
4. Click the folder icon and select the `.txt` file you copied to the config folder.
   - *Note: Do not import the .wav file directly using the  "Convolution" plugin; use the "Include" method to load the .txt file,  which contains the correct channel mapping.*

## License & Credits

- **License:** MIT License.
- **Libraries:** This project utilizes `pysofaconventions` (ANDERA) and `scipy/numpy`.

------

# wav-from-sofa (中文版)

**wav-from-sofa** 是一个专业的 Python 工具，用于从 SOFA 文件中提取 HRTF（头相关传输函数）参数，并生成高精度的 WAV 卷积文件。

它的设计初衷是为了在耳机上实现**虚拟环绕声**，通过模拟一对放置在听众正前方等边三角形位置（方位角 ±30°，边长 3 米）的高保真音箱，带来真实的听音室体验。

本程序导出的.wav卷积文件采用了 **HeSuVi 标准通道顺序 [LL, RR, LR, RL]**，因此可以有效地与EqualizerAPO配合使用。（具体导入EqualizerAPO的配置方法请见下文）

## 功能特性

- **标准立体声模拟：** 提取 ±30° 方位角、0° 仰角的音箱 HRTF 数据。
- **物理级渲染：** 完整保留 SOFA 数据中内嵌的自然双耳时间差（ITD），不引入人为的虚假延迟。
- **高精度重采样：** 支持导出为 44.1kHz, 48kHz, 88.2kHz, 96kHz, 176.4kHz 和 192kHz。
- **HeSuVi 兼容性：** 生成符合 True Stereo 虚拟化所需的 4 通道 WAV 文件结构。
- **自动配置：** 自动生成可直接用于 Equalizer APO 的配置文件（`.txt`）。

## 依赖库

本工具需要 Python 3 环境以及以下库：

- `pysofaconventions`
- `netCDF4`
- `numpy`
- `scipy`

## 安装方法

1. 确保已安装 Python。
2. 使用 pip 安装所需库：

bash

```bash
pip install pysofaconventions netCDF4 numpy scipy
```

## 使用方法

1. 运行脚本：

   bash

   ```bash
   python wav-from-sofa-v1.6.py
   ```

1. **第1步：** 将你的 `.sofa` 文件拖入终端窗口（或粘贴路径），按回车。
2. **第2步：** 脚本将验证文件并检测坐标约定。
3. **第3步：** 选择你需要的输出采样率（例如 48000 Hz）。
4. **第4步：** 输入保存 `.wav` 文件的路径（例如 `D:\HRTF\my_preset.wav`）。
5. **结果：** 脚本将生成两个文件：
   - `my_preset.wav` (卷积数据文件)
   - `my_preset.wav.txt` (Equalizer APO 配置文件)

## Equalizer APO 导入与配置指南（重要）

为了确保最佳听感并避免常见问题（如无声、爆音或权限错误），请务必严格按照以下步骤操作：

### 1. Windows 采样率匹配

**关键：** 生成的 WAV 文件采样率**必须**与你在 Windows 中设置的播放设备采样率一致。

- 进入 **控制面板 -> 声音 -> 播放**。
- 右键点击你的设备（DAC/耳机） -> **属性 -> 高级**。
- 确保“默认格式”与你在脚本中选择的采样率一致（例如 24位，48000 Hz）。

### 2. 文件位置与权限

Equalizer APO 通常没有权限读取安装目录以外的文件，这会导致卷积无法加载。

- **操作：** 请将生成的 `.wav` 文件**和** `.txt` 文件**同时复制**到 Equalizer APO 的 config 文件夹内。
- **默认路径：** `C:\Program Files\EqualizerAPO\config\`

### 3. 修改配置文件路径

由于你移动了文件位置，你需要检查并修改生成的 `.txt` 文件中的路径。

- 用记事本打开 `.txt` 文件。
- 找到以 `Convolution:` 开头的那一行。
- **修改前：** `Convolution: D:\Original\Path\my_preset.wav` (这是生成时的原始路径)
- **修改后：** `Convolution: my_preset.wav` (相对路径) **或者** `Convolution: C:\Program Files\EqualizerAPO\config\my_preset.wav` (绝对路径)。

### 4. 在 Editor 中导入

1. 打开 **Configuration Editor.exe**。
2. **添加前置放大 (Preamp)：** 在配置的最顶端添加一个 Preamp，并根据EqualizerAPO的“分析面板”内的**“峰值增益**”设置为一个负数，使“峰值增益”**不再是红字**。这是为了防止削波（Clipping）和爆音，因为卷积运算通常会增加整体增益。
3. **导入配置：** 点击绿色加号 -> Control -> **Include**。
4. 点击文件夹图标，选择你刚才复制到 config 文件夹内的 `.txt` 文件。
   - *注意：不要直接使用 "Convolution" 插件导入 .wav 文件，请使用 "Include" 导入 .txt 文件，因为 .txt 文件中包含了正确的通道矩阵映射信息。*

## 协议与致谢

- **开源协议：** MIT License.
- **引用库：** 本项目使用了 `pysofaconventions` (ANDERA) 以及 `scipy/numpy` 进行科学计算。