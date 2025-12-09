#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wav-from-sofa
从SOFA文件提取HRTF参数并生成WAV卷积文件

功能：提取"摆放在自己正前方的呈60度夹角、边长3米的等边三角形的一对音箱"的HRTF参数
并导出为指定采样率的WAV卷积文件，同时生成Equalizer APO配置文件。

依赖库：
    - pysofaconventions
    - numpy
    - scipy
    
作者：tunwen0
版本：1.6.0 
"""

import os
import sys
import warnings
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.spatial import cKDTree

# 抑制警告
warnings.filterwarnings('ignore')

# ============================================================
# 预设参数（固定值）
# ============================================================
INTERAURAL_DISTANCE = 0.15
SPEAKER_ANGLE = 30.0
SPEAKER_DISTANCE = 3.0
SUPPORTED_SAMPLE_RATES = [44100, 48000, 88200, 96000, 176400, 192000]
VERSION = "1.6.0"

# ============================================================
# 库导入检查
# ============================================================
try:
    from pysofaconventions import SOFAFile
except ImportError:
    print("=" * 60)
    print("错误：未找到 pysofaconventions 库")
    print("请使用以下命令安装：")
    print("    pip install pysofaconventions")
    print("=" * 60)
    sys.exit(1)

try:
    import netCDF4
except ImportError:
    print("=" * 60)
    print("错误：未找到 netCDF4 库")
    print("请使用以下命令安装：")
    print("    pip install netCDF4")
    print("=" * 60)
    sys.exit(1)


def clear_screen():
    """清屏函数"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """打印程序头部"""
    print("=" * 60)
    print(f"       wav-from-sofa v{VERSION} - SOFA转WAV卷积文件工具")
    print("=" * 60)
    print()


def spherical_to_cartesian(azimuth_deg, elevation_deg, distance):
    az_rad = np.radians(azimuth_deg)
    el_rad = np.radians(elevation_deg)
    x = distance * np.cos(el_rad) * np.cos(az_rad)
    y = distance * np.cos(el_rad) * np.sin(az_rad)
    z = distance * np.sin(el_rad)
    return np.array([x, y, z])


def cartesian_to_spherical(x, y, z):
    distance = np.sqrt(x**2 + y**2 + z**2)
    if distance < 1e-10:
        return 0.0, 0.0, 0.0
    elevation_deg = np.degrees(np.arcsin(np.clip(z / distance, -1.0, 1.0)))
    azimuth_deg = np.degrees(np.arctan2(y, x))
    return azimuth_deg, elevation_deg, distance


def load_sofa_file(filepath):
    try:
        sofa = SOFAFile(filepath, 'r')
        return sofa
    except Exception as e:
        print(f"错误：无法打开SOFA文件")
        print(f"详细信息：{e}")
        return None


def validate_sofa_file(sofa):
    try:
        data_ir = sofa.getDataIR()
        if data_ir is None or data_ir.size == 0:
            return False, "SOFA文件不包含脉冲响应数据(Data.IR)"
        if len(data_ir.shape) < 2:
            return False, "SOFA文件数据维度不正确"
        sr = sofa.getSamplingRate()
        if sr is None or (isinstance(sr, np.ndarray) and sr.size == 0):
            return False, "SOFA文件不包含采样率信息"
        positions = sofa.getSourcePositionValues()
        if positions is None or positions.size == 0:
            return False, "SOFA文件不包含源位置信息"
        return True, "验证通过"
    except Exception as e:
        return False, f"验证过程中出错：{e}"


def calculate_rms(data):
    return np.sqrt(np.mean(data ** 2))


def get_sofa_info(sofa):
    info = {}
    try:
        pos_units, coord_type = sofa.getSourcePositionInfo()
        info['coord_type'] = coord_type if coord_type else 'spherical'
        info['pos_units'] = pos_units if pos_units else 'degree, degree, metre'
    except:
        info['coord_type'] = 'spherical'
        info['pos_units'] = 'degree, degree, metre'
    
    try:
        sr = sofa.getSamplingRate()
        if isinstance(sr, np.ndarray):
            sr = float(sr.flatten()[0])
        info['sample_rate'] = int(sr)
    except:
        info['sample_rate'] = 48000
    
    try:
        sr_units = sofa.getSamplingRateUnits()
        info['sample_rate_units'] = sr_units if sr_units else 'hertz'
    except:
        info['sample_rate_units'] = 'hertz'
    
    try:
        positions = sofa.getSourcePositionValues()
        if len(positions.shape) == 1:
            info['num_measurements'] = 1
        else:
            info['num_measurements'] = positions.shape[0]
    except:
        info['num_measurements'] = 0
    
    try:
        data_ir = sofa.getDataIR()
        info['ir_length'] = data_ir.shape[-1]
        info['num_receivers'] = data_ir.shape[1] if len(data_ir.shape) >= 3 else 1
    except:
        info['ir_length'] = 0
        info['num_receivers'] = 2
    
    return info


def get_source_positions(sofa, info):
    positions_orig = sofa.getSourcePositionValues()
    if len(positions_orig.shape) == 1:
        positions_orig = positions_orig.reshape(1, -1)
    
    coord_type = info['coord_type'].lower() if info['coord_type'] else 'spherical'
    
    if 'cartesian' in coord_type:
        positions_cartesian = positions_orig.copy()
        positions_spherical = np.array([
            cartesian_to_spherical(pos[0], pos[1], pos[2])
            for pos in positions_orig
        ])
    else:
        positions_spherical = positions_orig.copy()
        positions_cartesian = np.array([
            spherical_to_cartesian(pos[0], pos[1], pos[2])
            for pos in positions_orig
        ])
    
    return positions_spherical, positions_cartesian


def find_nearest_measurements(positions_cartesian, target_cartesian, k=4):
    tree = cKDTree(positions_cartesian)
    k_actual = min(k, len(positions_cartesian))
    distances, indices = tree.query(target_cartesian, k=k_actual)
    if np.isscalar(distances):
        distances = np.array([distances])
        indices = np.array([indices])
    return distances, indices


def interpolate_hrtf_idw(data_ir, indices, distances, power=2):
    if distances[0] < 0.0001:
        return data_ir[indices[0]].copy()
    eps = 1e-10
    weights = 1.0 / (np.power(distances, power) + eps)
    weights = weights / np.sum(weights)
    ref_data = data_ir[indices[0]]
    result = np.zeros_like(ref_data, dtype=np.float64)
    for i, idx in enumerate(indices):
        result += weights[i] * data_ir[idx].astype(np.float64)
    return result


def extract_hrtf_for_direction_raw(sofa, info, target_azimuth, target_elevation, 
                                    target_distance=None):
    data_ir = sofa.getDataIR()
    if len(data_ir.shape) == 2:
        data_ir = data_ir[:, np.newaxis, :]
    
    positions_spherical, positions_cartesian = get_source_positions(sofa, info)
    
    if target_distance is None:
        target_distance = np.median(positions_spherical[:, 2])
    
    target_cartesian = spherical_to_cartesian(target_azimuth, target_elevation, 
                                               target_distance)
    distances, indices = find_nearest_measurements(positions_cartesian, 
                                                    target_cartesian, k=8)
    hrtf = interpolate_hrtf_idw(data_ir, indices, distances, power=2)
    return hrtf


def detect_conventions(sofa, info):
    positions_spherical, _ = get_source_positions(sofa, info)
    azimuths = positions_spherical[:, 0]
    elevations = positions_spherical[:, 1]
    
    horizontal_mask = np.abs(elevations) < 30
    if np.sum(horizontal_mask) > 0:
        horizontal_azimuths = azimuths[horizontal_mask]
    else:
        horizontal_azimuths = azimuths
    
    abs_azimuths = np.abs(horizontal_azimuths)
    max_abs_idx = np.argmax(abs_azimuths)
    test_azimuth = horizontal_azimuths[max_abs_idx]
    
    if test_azimuth < 0:
        test_azimuth = -test_azimuth
    
    test_azimuth = min(test_azimuth, 90.0)
    test_azimuth = max(test_azimuth, 30.0)
    
    print(f"  [检测] 使用方位角 {test_azimuth:.1f}° 进行约定检测...")
    
    hrtf_positive = extract_hrtf_for_direction_raw(sofa, info, +test_azimuth, 0.0)
    if len(hrtf_positive.shape) == 1:
        hrtf_positive = np.vstack([hrtf_positive, hrtf_positive])
    
    receiver0_rms = calculate_rms(hrtf_positive[0])
    receiver1_rms = calculate_rms(hrtf_positive[1])
    
    hrtf_negative = extract_hrtf_for_direction_raw(sofa, info, -test_azimuth, 0.0)
    if len(hrtf_negative.shape) == 1:
        hrtf_negative = np.vstack([hrtf_negative, hrtf_negative])
    
    neg_receiver0_rms = calculate_rms(hrtf_negative[0])
    neg_receiver1_rms = calculate_rms(hrtf_negative[1])
    
    pos_stronger = 0 if receiver0_rms > receiver1_rms else 1
    neg_stronger = 0 if neg_receiver0_rms > neg_receiver1_rms else 1
    
    if pos_stronger != neg_stronger:
        positive_side_receiver = pos_stronger
        
        try:
            receiver_pos = sofa.getReceiverPositionValues()
            if receiver_pos is not None and len(receiver_pos) >= 2:
                y0 = receiver_pos[0][1] if len(receiver_pos[0]) > 1 else 0
                y1 = receiver_pos[1][1] if len(receiver_pos[1]) > 1 else 0
                
                if abs(y0 - y1) > 0.001:
                    if y0 > y1:
                        left_ear_idx = 0
                        right_ear_idx = 1
                    else:
                        left_ear_idx = 1
                        right_ear_idx = 0
                    
                    if positive_side_receiver == left_ear_idx:
                        azimuth_sign = 1
                    else:
                        azimuth_sign = -1
                    
                    print(f"    基于接收器位置：左耳=接收器{left_ear_idx}，右耳=接收器{right_ear_idx}")
                    print(f"    方位角约定：正值向{'左' if azimuth_sign == 1 else '右'}")
                    return azimuth_sign, left_ear_idx, right_ear_idx
        except:
            pass
        
        left_ear_idx = 0
        right_ear_idx = 1
        
        if positive_side_receiver == right_ear_idx:
            azimuth_sign = -1
        else:
            azimuth_sign = 1
        
        print(f"    使用默认假设：左耳=接收器0，右耳=接收器1")
        print(f"    方位角约定：正值向{'左' if azimuth_sign == 1 else '右'}")
        
        return azimuth_sign, left_ear_idx, right_ear_idx
    
    else:
        print("  [警告] 无法确定约定，使用默认值")
        return -1, 0, 1


def extract_stereo_speaker_hrtf(sofa, info, azimuth_sign, left_ear_idx, right_ear_idx):
    left_speaker_azimuth = SPEAKER_ANGLE * azimuth_sign
    right_speaker_azimuth = -SPEAKER_ANGLE * azimuth_sign
    
    print(f"  实际提取方位角：左音箱={left_speaker_azimuth:+.1f}°，右音箱={right_speaker_azimuth:+.1f}°")
    
    left_hrtf_raw = extract_hrtf_for_direction_raw(
        sofa, info, left_speaker_azimuth, 0.0, SPEAKER_DISTANCE
    )
    right_hrtf_raw = extract_hrtf_for_direction_raw(
        sofa, info, right_speaker_azimuth, 0.0, SPEAKER_DISTANCE
    )
    
    if len(left_hrtf_raw.shape) == 1:
        left_hrtf_raw = np.vstack([left_hrtf_raw, left_hrtf_raw])
    if len(right_hrtf_raw.shape) == 1:
        right_hrtf_raw = np.vstack([right_hrtf_raw, right_hrtf_raw])
    
    left_speaker_hrtf = np.vstack([
        left_hrtf_raw[left_ear_idx],
        left_hrtf_raw[right_ear_idx]
    ])
    right_speaker_hrtf = np.vstack([
        right_hrtf_raw[left_ear_idx],
        right_hrtf_raw[right_ear_idx]
    ])
    
    return left_speaker_hrtf, right_speaker_hrtf


def resample_signal(signal_data, original_sr, target_sr):
    if original_sr == target_sr:
        return signal_data.copy()
    duration = len(signal_data) / original_sr
    new_length = int(np.round(duration * target_sr))
    resampled = signal.resample(signal_data, new_length)
    return resampled


def generate_eqapo_config(wav_path):
    """
    生成Equalizer APO的配置文件
    使用 HeSuVi 通道顺序 [LL, RR, LR, RL] 对应的矩阵
    并使用绝对路径以确保文件被正确加载
    """
    config_path = wav_path + ".txt"
    wav_filename = os.path.basename(wav_path)
    
    # 获取绝对路径并转为正斜杠（EqAPO兼容性更好）
    abs_wav_path = os.path.abspath(wav_path).replace('\\', '/')
    
    config_content = f"""# Equalizer APO Configuration for {wav_filename}
# Generated by wav-from-sofa v{VERSION}
# 
# 使用方法：
# 1. 打开 Equalizer APO 的 Configuration Editor
# 2. 添加 "Include" (Control -> Include)
# 3. 选择此文件：{os.path.basename(config_path)}

Convolution: {abs_wav_path}
    0 0 0
    1 1 1
    0 2 1
    1 3 0

# 映射说明 (InputIndex ImpulseIndex OutputIndex):
# 基于 HeSuVi 标准通道顺序 [LL, RR, LR, RL]
#
# 0 0 0 : 输入左(0) 卷积 WAV通道1(LL) -> 输出左(0) [左音箱->左耳]
# 1 1 1 : 输入右(1) 卷积 WAV通道2(RR) -> 输出右(1) [右音箱->右耳]
# 0 2 1 : 输入左(0) 卷积 WAV通道3(LR) -> 输出右(1) [左音箱->右耳]
# 1 3 0 : 输入右(1) 卷积 WAV通道4(RL) -> 输出左(0) [右音箱->左耳]
#
"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        return True, config_path
    except Exception as e:
        return False, str(e)


def create_convolution_wav(left_speaker_hrtf, right_speaker_hrtf,
                           original_sr, target_sr, output_path):
    """
    创建卷积WAV文件
    使用 HeSuVi 标准顺序 [LL, RR, LR, RL]
    """
    ls_to_left_ear = left_speaker_hrtf[0].astype(np.float64)
    ls_to_right_ear = left_speaker_hrtf[1].astype(np.float64)
    rs_to_left_ear = right_speaker_hrtf[0].astype(np.float64)
    rs_to_right_ear = right_speaker_hrtf[1].astype(np.float64)
    
    print()
    print("  诊断信息 - 各通道RMS能量：")
    print(f"    左音箱→左耳 (LL): {calculate_rms(ls_to_left_ear):.6f}")
    print(f"    右音箱→右耳 (RR): {calculate_rms(rs_to_right_ear):.6f}")
    print(f"    左音箱→右耳 (LR): {calculate_rms(ls_to_right_ear):.6f}")
    print(f"    右音箱→左耳 (RL): {calculate_rms(rs_to_left_ear):.6f}")
    
    # 重采样
    ls_to_left_ear = resample_signal(ls_to_left_ear, original_sr, target_sr)
    ls_to_right_ear = resample_signal(ls_to_right_ear, original_sr, target_sr)
    rs_to_left_ear = resample_signal(rs_to_left_ear, original_sr, target_sr)
    rs_to_right_ear = resample_signal(rs_to_right_ear, original_sr, target_sr)
    
    # 统一长度
    max_length = max(len(ls_to_left_ear), len(ls_to_right_ear),
                     len(rs_to_left_ear), len(rs_to_right_ear))
    
    ls_to_left_ear = np.pad(ls_to_left_ear, (0, max_length - len(ls_to_left_ear)))
    ls_to_right_ear = np.pad(ls_to_right_ear, (0, max_length - len(ls_to_right_ear)))
    rs_to_left_ear = np.pad(rs_to_left_ear, (0, max_length - len(rs_to_left_ear)))
    rs_to_right_ear = np.pad(rs_to_right_ear, (0, max_length - len(rs_to_right_ear)))
    
    # 组合数据：HeSuVi 顺序 [LL, RR, LR, RL]
    output_data = np.column_stack([
        ls_to_left_ear,   # 通道1: L → L (主)
        rs_to_right_ear,  # 通道2: R → R (主)  <-- 放在这里以防矩阵失效
        ls_to_right_ear,  # 通道3: L → R (串扰)
        rs_to_left_ear    # 通道4: R → L (串扰)
    ])
    
    # 归一化
    max_amplitude = np.max(np.abs(output_data))
    if max_amplitude > 0:
        output_data = output_data / max_amplitude * 0.95
    
    output_data = output_data.astype(np.float32)
    
    # 保存WAV
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    wavfile.write(output_path, target_sr, output_data)
    
    # 生成配置文件
    return generate_eqapo_config(output_path)


def get_available_sample_rates(original_sr):
    return SUPPORTED_SAMPLE_RATES.copy()


def main():
    while True:
        clear_screen()
        print_header()
        
        # 第一步：输入SOFA文件路径
        print("【第1步】请输入您的sofa文件的储存路径，然后按回车：")
        print("（例如：C:\\Users\\Username\\Desktop\\myhrtf.sofa）")
        print()
        
        sofa_path = input(">>> ").strip().strip('"').strip("'")
        
        if not sofa_path:
            print("\n错误：请输入有效的文件路径")
            input("\n按回车键继续...")
            continue
        
        if not os.path.exists(sofa_path):
            print(f"\n错误：文件不存在\n路径：{sofa_path}")
            input("\n按回车键继续...")
            continue
        
        # 第二步：加载和验证
        print("\n【第2步】正在加载和验证SOFA文件...")
        
        sofa = load_sofa_file(sofa_path)
        if sofa is None:
            input("\n按回车键继续...")
            continue
        
        try:
            is_valid, validation_msg = validate_sofa_file(sofa)
            if not is_valid:
                print(f"错误：{validation_msg}")
                input("\n按回车键继续...")
                sofa.close()
                continue
            
            info = get_sofa_info(sofa)
            print(f"  ✓ 坐标系类型：{info['coord_type']}")
            print(f"  ✓ 原始采样率：{info['sample_rate']} Hz")
            print("\n  验证完成，SOFA文件有效。")
            
        except Exception as e:
            print(f"错误：验证SOFA文件时出错 - {e}")
            input("\n按回车键继续...")
            try:
                sofa.close()
            except:
                pass
            continue
        
        # 第三步：检测约定并提取HRTF
        print("\n【第3步】正在检测约定并提取HRTF参数...")
        print()
        
        try:
            azimuth_sign, left_ear_idx, right_ear_idx = detect_conventions(sofa, info)
            
            left_hrtf, right_hrtf = extract_stereo_speaker_hrtf(
                sofa, info, azimuth_sign, left_ear_idx, right_ear_idx
            )
            print("\n  ✓ HRTF参数提取完成！")
            
        except Exception as e:
            print(f"错误：提取HRTF参数失败 - {e}")
            import traceback
            traceback.print_exc()
            input("\n按回车键继续...")
            try:
                sofa.close()
            except:
                pass
            continue
        
        # 第四步：选择采样率
        print()
        available_rates = get_available_sample_rates(info['sample_rate'])
        
        print("【第4步】请选择输出采样率：")
        print()
        
        for i, rate in enumerate(available_rates, 1):
            if rate == info['sample_rate']:
                print(f"  {i}: {rate} Hz  ← 原始采样率（推荐）")
            else:
                print(f"  {i}: {rate} Hz")
        print()
        
        try:
            choice_num = int(input(">>> ").strip())
            if choice_num < 1 or choice_num > len(available_rates):
                raise ValueError()
            target_sr = available_rates[choice_num - 1]
            print(f"\n  已选择采样率：{target_sr} Hz")
        except ValueError:
            print("\n错误：请输入有效的数字")
            input("\n按回车键继续...")
            try:
                sofa.close()
            except:
                pass
            continue
        
        # 第五步：输入输出路径
        print("\n【第5步】请输入您的wav卷积文件的储存路径，然后按回车：")
        print("（例如：D:\\Files\\myhrtf.wav）")
        print()
        
        wav_path = input(">>> ").strip().strip('"').strip("'")
        
        if not wav_path:
            print("\n错误：请输入有效的文件路径")
            input("\n按回车键继续...")
            try:
                sofa.close()
            except:
                pass
            continue
        
        if not wav_path.lower().endswith('.wav'):
            wav_path += '.wav'
        
        # 第六步：生成
        print("\n【第6步】正在生成WAV卷积文件...")
        print()
        print(f"  - 目标采样率：{target_sr} Hz")
        print(f"  - 输出格式：4通道 32位浮点 WAV")
        print(f"  - 通道顺序：[LL, RR, LR, RL] (HeSuVi标准)")
        print(f"  - 输出路径：{wav_path}")
        
        try:
            success, config_path = create_convolution_wav(
                left_hrtf, right_hrtf,
                info['sample_rate'], target_sr,
                wav_path
            )
            
            try:
                sofa.close()
            except:
                pass
            
            print()
            print("=" * 60)
            print()
            print("  ★ 导出成功！")
            print()
            print(f"  WAV文件：{wav_path}")
            if success:
                print(f"  配置文件：{config_path}")
                print()
                print("  【重要提示】")
                print("  请将导出的txt和wav文件都复制进EqualizerAPO的config文件夹内")
                print("  然后将txt文件内的Convolution路径手动修改为config文件夹内的wav文件的路径")
                print("  然后在 Equalizer APO 中使用 'Include' 命令加载生成的 .txt 文件。")
                print("  注意只加载txt文件就好，不要在EqualizerAPO里直接加载wav卷积文件")
            print()
            print("=" * 60)
            
        except Exception as e:
            print(f"\n错误：导出失败 - {e}")
            import traceback
            traceback.print_exc()
            input("\n按回车键继续...")
            try:
                sofa.close()
            except:
                pass
            continue
        
        # 询问继续
        print()
        print("导出成功，继续处理新的sofa文件请按1，退出程序请按2：")
        print()
        
        try:
            choice = input(">>> ").strip()
            if choice == "2":
                print("\n感谢使用 wav-from-sofa，再见！\n")
                break
        except (KeyboardInterrupt, EOFError):
            print("\n程序已退出。")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断，已退出。")
        sys.exit(0)
    except Exception as e:
        print(f"\n程序发生未预期的错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)