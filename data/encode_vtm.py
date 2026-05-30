from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def encode_decode_vtm(
    encoder_app: str,
    decoder_app: str,
    cfg_path: str,
    input_yuv: str,
    width: int,
    height: int,
    frames: int,
    qp: int,
    bitstream_path: str,
    recon_yuv: str,
) -> None:
    """Minimal VTM command wrapper.

    You must prepare YUV input externally. This wrapper is intentionally thin so
    it remains compatible with local VTM builds.
    """
    enc_cmd = [
        encoder_app,
        "-c", cfg_path,
        "-i", input_yuv,
        "-b", bitstream_path,
        "-o", recon_yuv,
        "-wdt", str(width),
        "-hgt", str(height),
        "-f", str(frames),
        "-q", str(qp),
    ]
    run_cmd(enc_cmd)
    dec_cmd = [decoder_app, "-b", bitstream_path, "-o", recon_yuv]
    run_cmd(dec_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--input-yuv", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--qp", type=int, required=True)
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--recon-yuv", required=True)
    args = parser.parse_args()
    Path(args.bitstream).parent.mkdir(parents=True, exist_ok=True)
    Path(args.recon_yuv).parent.mkdir(parents=True, exist_ok=True)
    encode_decode_vtm(
        args.encoder,
        args.decoder,
        args.cfg,
        args.input_yuv,
        args.width,
        args.height,
        args.frames,
        args.qp,
        args.bitstream,
        args.recon_yuv,
    )
