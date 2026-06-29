"""配置数据源认证凭据。

运行方式：
  python scripts/setup_credentials.py

交互式引导你配置三个数据平台的认证：
1. NASA Earthdata (下载 NSIDC SIC/SIT, ICESat-2)
2. Copernicus CDS (下载 ERA5 再分析)
3. Copernicus Marine (下载海冰再分析)

凭据保存在用户主目录下，不会提交到 git。
"""

from __future__ import annotations

import os
from pathlib import Path
import getpass


def setup_earthdata():
    """配置 NASA Earthdata 登录。"""
    print("\n" + "=" * 60)
    print("1. NASA Earthdata (NSIDC SIC, CryoSat-2, ICESat-2)")
    print("=" * 60)
    print("注册地址: https://urs.earthdata.nasa.gov/")
    print("注册后在 'My Profile' -> 'User Token' 获取用户名和密码\n")

    username = input("Earthdata 用户名 (回车跳过): ").strip()
    if not username:
        print("  跳过 Earthdata 配置")
        return False

    password = getpass.getpass("Earthdata 密码: ")

    netrc_path = Path.home() / ".netrc"
    # 追加或更新
    lines = []
    if netrc_path.exists():
        lines = netrc_path.read_text().splitlines()
        # 移除已有的 urs.earthdata.nasa.gov 条目
        new_lines = []
        skip = False
        for line in lines:
            if "urs.earthdata.nasa.gov" in line:
                skip = True
                continue
            if skip and (line.strip().startswith("login") or line.strip().startswith("password")):
                continue
            if skip and not line.strip():
                skip = False
                continue
            new_lines.append(line)
        lines = new_lines

    lines.append(f"machine urs.earthdata.nasa.gov")
    lines.append(f"  login {username}")
    lines.append(f"  password {password}")
    lines.append("")

    netrc_path.write_text("\n".join(lines))
    netrc_path.chmod(0o600)
    print(f"  已保存到 {netrc_path}")

    # 创建 .cookies 目录 (earthaccess 需要)
    cookie_dir = Path.home() / ".earthdata" / "cookies"
    cookie_dir.mkdir(parents=True, exist_ok=True)

    return True


def setup_cds():
    """配置 Copernicus CDS API key。"""
    print("\n" + "=" * 60)
    print("2. Copernicus CDS (ERA5 再分析数据)")
    print("=" * 60)
    print("注册地址: https://cds.climate.copernicus.eu/")
    print("注册后在 https://cds.climate.copernicus.eu/ how-to-api 获取 API key\n")

    uid = input("CDS UID (数字, 回车跳过): ").strip()
    if not uid:
        print("  跳过 CDS 配置")
        return False

    api_key = getpass.getpass("CDS API Key: ")

    cdsapirc = Path.home() / ".cdsapirc"
    content = f"url: https://cds.climate.copernicus.eu/api\nkey: {uid}:{api_key}\n"
    cdsapirc.write_text(content)
    cdsapirc.chmod(0o600)
    print(f"  已保存到 {cdsapirc}")
    return True


def setup_copernicus_marine():
    """配置 Copernicus Marine 账号。"""
    print("\n" + "=" * 60)
    print("3. Copernicus Marine (海冰再分析数据)")
    print("=" * 60)
    print("注册地址: https://marine.copernicus.eu/\n")

    username = input("Copernicus Marine 用户名 (回车跳过): ").strip()
    if not username:
        print("  跳过 Copernicus Marine 配置")
        return False

    password = getpass.getpass("Copernicus Marine 密码: ")

    # copernicusmarine 使用环境变量或配置文件
    # 创建配置文件
    config_dir = Path.home() / ".copernicusmarine"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "copernicusmarine-datastore-credentials.ini"
    content = f"[default]\nusername = {username}\npassword = {password}\n"
    config_file.write_text(content)
    config_file.chmod(0o600)
    print(f"  已保存到 {config_file}")
    return True


def verify_credentials():
    """验证凭据是否可用。"""
    print("\n" + "=" * 60)
    print("验证凭据")
    print("=" * 60)

    # Earthdata
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text():
        print("✓ NASA Earthdata 凭据已配置")
        try:
            import earthaccess
            auth = earthaccess.login()
            print(f"  登录成功: {auth}")
        except Exception as e:
            print(f"  登录验证失败: {e}")
    else:
        print("✗ NASA Earthdata 凭据未配置")

    # CDS
    cdsapirc = Path.home() / ".cdsapirc"
    if cdsapirc.exists():
        print("✓ CDS API key 已配置")
    else:
        print("✗ CDS API key 未配置")

    # Copernicus Marine
    cm_config = Path.home() / ".copernicusmarine" / "copernicusmarine-datastore-credentials.ini"
    if cm_config.exists():
        print("✓ Copernicus Marine 凭据已配置")
    else:
        print("✗ Copernicus Marine 凭据未配置")


def main():
    print("数据源认证凭据配置向导")
    print("本脚本帮助你配置三个数据平台的认证。")
    print("凭据保存在用户主目录下，不会提交到 git。\n")

    setup_earthdata()
    setup_cds()
    setup_copernicus_marine()
    verify_credentials()

    print("\n" + "=" * 60)
    print("配置完成！接下来运行下载脚本：")
    print("  python scripts/download_data.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
