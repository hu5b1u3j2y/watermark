import cv2
import numpy as np
import argparse

from watermark import embed, extract
from attacks import gaussian_noise, jpeg_compression
from metrics import compute_psnr, compute_ber

np.random.seed(42)

def main():
    parser = argparse.ArgumentParser(description="QIM Watermark System (CLI)")

    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default="watermarked.png", help="Output image path")
    parser.add_argument("--attack_noise", action="store_true", help="Apply Gaussian noise")
    parser.add_argument("--attack_jpeg", action="store_true", help="Apply JPEG compression")

    args = parser.parse_args()

    image = cv2.imread(args.input, 0)
    if image is None:
        print("Erreur: impossible de lire l'image")
        return

    print("Image chargée")

    watermark = np.random.randint(0, 2, 500)
    print("Watermark généré")

    watermarked = embed(image, watermark)
    print("Watermark intégré")

    cv2.imwrite(args.output, watermarked)
    print(f"Image sauvegardée: {args.output}")

    attacked = watermarked.copy()

    if args.attack_noise:
        attacked = gaussian_noise(attacked)
        print("Attaque: bruit gaussien appliqué")

    if args.attack_jpeg:
        attacked = jpeg_compression(attacked)
        print("Attaque: compression JPEG appliquée")

    psnr = compute_psnr(image, watermarked)

    if args.attack_noise or args.attack_jpeg:
        extracted = extract(attacked, len(watermark))
        ber = compute_ber(watermark, extracted)
        print(f"PSNR: {psnr:.2f} | BER: {ber:.4f}")
    else:
        print(f"PSNR: {psnr:.2f}")


if __name__ == "__main__":
    main()
