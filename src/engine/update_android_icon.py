import os
from pathlib import Path
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
LOGO_PATH = WORKSPACE_ROOT / "web_app" / "logo_icon.png"
RES_DIR = WORKSPACE_ROOT / "android" / "app" / "src" / "main" / "res"

# Android Mipmap dimensions
RESOLUTIONS = {
    "mipmap-mdpi": 48,
    "mipmap-hdpi": 72,
    "mipmap-xhdpi": 96,
    "mipmap-xxhdpi": 144,
    "mipmap-xxxhdpi": 192
}

def generate_icons():
    if not LOGO_PATH.exists():
        print(f"Error: {LOGO_PATH} not found!")
        return

    orig_img = Image.open(LOGO_PATH).convert("RGBA")

    for folder_name, size in RESOLUTIONS.items():
        folder_path = RES_DIR / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        # Create square icon with padding and subtle dark background
        bg = Image.new("RGBA", (size, size), (15, 23, 42, 255))
        
        # Scale logo into padded canvas (80% of total box)
        pad_size = int(size * 0.75)
        resized_logo = orig_img.resize((pad_size, pad_size), Image.Resampling.LANCZOS)

        # Center logo
        offset = ((size - pad_size) // 2, (size - pad_size) // 2)
        bg.paste(resized_logo, offset, resized_logo)

        # Save as ic_launcher.png, ic_launcher_round.png, ic_launcher_foreground.png
        bg.save(folder_path / "ic_launcher.png", "PNG")
        bg.save(folder_path / "ic_launcher_round.png", "PNG")

        # Foreground logo (transparent background)
        fg = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        fg.paste(resized_logo, offset, resized_logo)
        fg.save(folder_path / "ic_launcher_foreground.png", "PNG")

        print(f"Generated icons for {folder_name} ({size}x{size}px)")

    # Also update web app favicon
    fav_path = WORKSPACE_ROOT / "web_app" / "favicon.ico"
    orig_img.resize((64, 64), Image.Resampling.LANCZOS).save(fav_path, format="ICO")
    print("Updated web_app/favicon.ico")

if __name__ == "__main__":
    generate_icons()
