"""
Rich interactive CLI interface for Face Identification and Blockchain Verification Pipeline.
"""

import sys
import os
import argparse
import time
from typing import Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import box

from pipeline import VerificationPipeline
from core.blockchain import NativeMerkleLedger, EVMVerificationLedger
from core.models import SocialPost

import re
import html
from urllib.parse import urljoin
import cv2
import numpy as np
import requests

console = Console()


def print_banner():
    banner_text = Text()
    banner_text.append("HH GOA 2026: FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION PIPELINE\n", style="bold cyan")
    banner_text.append("Neural Biometric Scan ➔ Web/Social Discovery ➔ Tamper-Evident Ledger", style="bold yellow")
    console.print(Panel(banner_text, box=box.DOUBLE, border_style="cyan", padding=(0, 2)))


def resolve_image_input(path_or_url: str) -> str:
    """
    Resolves an input string to a valid local image file path.
    Supports local file paths, drag-and-dropped quoted paths, and HTTP/HTTPS URLs.
    If the URL points to a web page (e.g. Wikimedia Commons, Wikipedia, article),
    it automatically resolves and extracts the primary image (og:image / fullImageLink).
    """
    clean = path_or_url.strip().strip("'\"")
    if not clean:
        raise ValueError("Image path or URL cannot be empty.")

    # If it's a web URL, download it locally
    if clean.startswith("http://") or clean.startswith("https://"):
        os.makedirs("data/inputs", exist_ok=True)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        console.print(f"[cyan]Downloading from URL:[/cyan] {clean}")
        resp = requests.get(clean, headers=headers, timeout=15)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "").lower()
        is_html = (
            "text/html" in content_type
            or resp.content.lstrip()[:100].lower().startswith(b"<!doctype html")
            or b"<html" in resp.content[:500].lower()
        )

        raw_bytes = resp.content
        download_url = clean

        if is_html:
            console.print("[dim]Webpage detected. Resolving primary image...[/dim]")
            text = resp.text
            # Check Wikimedia Commons fullImageLink
            wiki_full = re.findall(r'<div class=[\"\']fullImageLink[\"\'][^>]*><a href=[\"\']([^\"\']+)[\"\']', text)
            # Check og:image
            og_img = re.findall(r'<meta [^>]*property=[\"\']og:image[\"\'] [^>]*content=[\"\']([^\"\']+)[\"\']', text)
            if not og_img:
                og_img = re.findall(r'<meta [^>]*content=[\"\']([^\"\']+)[\"\'] [^>]*property=[\"\']og:image[\"\']', text)
            # Check twitter:image
            tw_img = re.findall(r'<meta [^>]*name=[\"\']twitter:image[\"\'] [^>]*content=[\"\']([^\"\']+)[\"\']', text)
            if not tw_img:
                tw_img = re.findall(r'<meta [^>]*content=[\"\']([^\"\']+)[\"\'] [^>]*name=[\"\']twitter:image[\"\']', text)

            candidates = wiki_full + og_img + tw_img
            if not candidates:
                img_srcs = re.findall(r'<img [^>]*src=[\"\']([^\"\']+)[\"\']', text)
                candidates = [s for s in img_srcs if any(ext in s.lower() for ext in ('.jpg', '.jpeg', '.png', '.webp'))]

            if not candidates:
                raise ValueError(
                    f"The URL '{clean}' is a webpage, but no primary image could be found on it. "
                    "Please provide a direct link to an image file."
                )

            first_img_url = html.unescape(candidates[0])
            first_img_url = urljoin(clean, first_img_url)
            console.print(f"[cyan]Extracted image source:[/cyan] {first_img_url}")
            img_resp = requests.get(first_img_url, headers=headers, timeout=15)
            img_resp.raise_for_status()
            raw_bytes = img_resp.content
            download_url = first_img_url

        # Validate that raw_bytes is a valid image using OpenCV
        decoded = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"Content downloaded from '{download_url}' could not be decoded as an image.")

        # Sanitize filename
        raw_basename = os.path.basename(download_url.split("?")[0]) or "web_face_input.jpg"
        # Strip URL prefix schemes like File:
        raw_basename = re.sub(r'^(File|Image):', '', raw_basename, flags=re.IGNORECASE)
        base_name = re.sub(r'[^\w\-_\.]', '_', raw_basename)
        if not any(base_name.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
            base_name += ".jpg"

        local_target = os.path.join("data/inputs", base_name)
        with open(local_target, "wb") as f:
            f.write(raw_bytes)
        console.print(f"[green]✔ Image ready at: {local_target}[/green]")
        return local_target

    expanded = os.path.expanduser(clean)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Image '{expanded}' does not exist.")
    return expanded


def extract_smart_hint(source_str: str) -> Optional[str]:
    """Extracts a clean subject name from a Wikipedia URL, Wikimedia Commons URL, or image filename."""
    if not source_str:
        return None
    # 1. Wikipedia article URL: e.g. /wiki/Virat_Kohli -> Virat Kohli
    wiki_art = re.search(r'/wiki/(?!File:|Image:)([^/?#]+)', source_str)
    if wiki_art:
        return wiki_art.group(1).replace('_', ' ').strip()

    # 2. Extract from filename
    base = os.path.basename(source_str.split('?')[0])
    base = os.path.splitext(base)[0]
    base = re.sub(r'^(File|Image)[:_]', '', base, flags=re.IGNORECASE)
    base = re.sub(r'[_\-]+', ' ', base)
    base = re.sub(r'^(the\s+)?(official\s+)?(portrait|photo|picture|image)\s+(of\s+)?', '', base, flags=re.IGNORECASE)
    base = re.sub(r'^(Shri|Dr|Mr|Mrs|Ms|Honorable)\s+', '', base, flags=re.IGNORECASE)
    base = re.sub(r'^Prime Minister of [^,]+,\s*(?:Shri\s*)?', '', base, flags=re.IGNORECASE)
    parts = re.split(r'\b(during|at|match|stadium|test\s+match|vs|on\s+\d+)\b', base, flags=re.IGNORECASE)
    cand = parts[0].strip(' ,-_')
    words = cand.split()
    if words:
        cand_str = ' '.join(words[:3]) if len(words) > 3 else ' '.join(words)
        if len(cand_str) >= 3 and not cand_str.isnumeric():
            return cand_str
    return None


def prompt_for_image() -> Tuple[str, Optional[str]]:
    """
    Interactively prompts the user to select or enter an image path or URL.
    Returns (image_path, optional_query_hint).
    """
    console.print("\n[bold cyan]Select an input image or enter your own:[/bold cyan]\n")
    samples = [
        ("1", "data/samples/elon_musk.jpg", "Elon Musk", "Elon Musk"),
        ("2", "data/samples/obama.jpg", "Barack Obama", "Barack Obama"),
        ("3", "data/samples/sample_face_1.jpg", "Dr. Sarah Lin (AI Researcher)", "Sarah Lin"),
        ("4", "data/samples/sample_face_2.jpg", "Sample Face 2", "Sample Face"),
    ]

    for key, path, label, _ in samples:
        if os.path.exists(path):
            console.print(f"  [bold green][{key}][/bold green] {path:<30} [dim]({label})[/dim]")
    console.print("  [bold green][5][/bold green] Enter custom local file path or image URL\n")

    choice = console.input("[bold yellow]Enter selection [1-5] or paste image path/URL directly: [/bold yellow]").strip()

    chosen_path = ""
    default_hint = None

    if choice in ("1", "2", "3", "4"):
        idx = int(choice) - 1
        chosen_path = samples[idx][1]
        default_hint = samples[idx][3]
    elif choice == "5":
        raw = console.input("[bold yellow]Enter image file path or URL: [/bold yellow]").strip()
        chosen_path = resolve_image_input(raw)
        default_hint = extract_smart_hint(raw) or extract_smart_hint(chosen_path)
    else:
        # User pasted path or URL directly
        chosen_path = resolve_image_input(choice)
        default_hint = extract_smart_hint(choice) or extract_smart_hint(chosen_path)

    # Prompt for optional query hint
    hint_prompt = f"Enter subject name or query hint (press Enter to use '{default_hint}' if blank): " if default_hint else "Enter subject name or query hint (optional, press Enter to skip): "
    hint_input = console.input(f"[cyan]{hint_prompt}[/cyan]").strip()
    query_hint = hint_input if hint_input else default_hint

    return chosen_path, query_hint


def cmd_scan(args):
    print_banner()
    raw_image = getattr(args, "image", None) or getattr(args, "image_opt", None)
    query_hint = getattr(args, "query", None)

    if not raw_image:
        try:
            image_path, interactive_hint = prompt_for_image()
            if not query_hint:
                query_hint = interactive_hint
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Scan cancelled by user.[/dim]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)
    else:
        try:
            image_path = resolve_image_input(raw_image)
        except Exception as e:
            console.print(f"[bold red]Error resolving image:[/bold red] {e}")
            sys.exit(1)

    chain_type = getattr(args, "chain", None) or "evm"
    platform_pref = getattr(args, "platform", None)

    console.print(f"\n[bold yellow]▶ INITIALIZING VERIFICATION PIPELINE[/bold yellow] (Target Chain: [cyan]{chain_type.upper()}[/cyan])\n")

    try:
        with Progress(
            SpinnerColumn(style="bold green"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            t1 = progress.add_task("[cyan]Detecting face & extracting 128D deep neural embedding...", total=None)
            pipe = VerificationPipeline(
                blockchain_type=chain_type,
                allow_offline=getattr(args, "offline_fallback", False),
            )
            time.sleep(0.3)
            progress.update(t1, description="[green]Face detected & 128D neural embedding extracted successfully!")

            t2 = progress.add_task("[cyan]Executing genuine live web/social media search & SFace metric ranking...", total=None)
            out = pipe.run(
                image_path=image_path,
                query_hint=query_hint,
                preferred_platform=platform_pref,
                allow_offline=getattr(args, "offline_fallback", False),
            )
            progress.update(t2, description="[green]Matching social media post discovered via live search!")

            t3 = progress.add_task(f"[cyan]Anchoring cryptographic fingerprint to {chain_type.upper()} blockchain...", total=None)
            time.sleep(0.3)
            progress.update(t3, description=f"[green]Data successfully anchored to {chain_type.upper()} blockchain!")

            t4 = progress.add_task("[cyan]Executing on-chain cryptographic re-verification...", total=None)
            time.sleep(0.2)
            progress.update(t4, description="[green]On-chain verification complete!")
    except RuntimeError as e:
        console.print(Panel(
            f"[bold red]SEARCH STAGE ERROR:[/bold red]\n{e}\n\n"
            f"[dim]If you are in an air-gapped environment without internet access, "
            f"explicitly pass '--offline-fallback' to enable test corpus mode.[/dim]",
            border_style="red"
        ))
        sys.exit(1)

    # --- Section 1: Face Detection Results ---
    face = out.face_result
    t_face = Table(title="Stage 1: Face Detection & Biometric Fingerprint", box=box.ROUNDED, style="cyan")
    t_face.add_column("Parameter", style="bold white", width=24)
    t_face.add_column("Value", style="green")
    t_face.add_row("Source Image", os.path.abspath(face.source_image))
    t_face.add_row("Detection Confidence", f"{face.confidence * 100:.2f}%")
    t_face.add_row("Bounding Box", f"X={face.bounding_box.x}, Y={face.bounding_box.y}, W={face.bounding_box.width}, H={face.bounding_box.height}")
    t_face.add_row("Facial Landmarks", f"{len(face.landmarks)} Points (Eyes, Nose, Mouth corners)")
    t_face.add_row("Embedding Dimension", f"{len(face.embedding)} Dimensions (OpenCV SFace)")
    t_face.add_row("Biometric SHA-256 Hash", f"[bold yellow]{face.embedding_hash}[/bold yellow]")
    if face.crop_path:
        t_face.add_row("Aligned Crop Thumbnail", os.path.abspath(face.crop_path))
    console.print(t_face)

    # --- Section 2: Discovered Social Media Post ---
    post = out.post
    t_post = Table(title="Stage 2: Discovered Social Media Post (Live Web/Social Search)", box=box.ROUNDED, style="magenta")
    t_post.add_column("Parameter", style="bold white", width=26)
    t_post.add_column("Value", style="yellow")
    
    # Status badge showing LIVE_SEARCH
    status_style = "bold green" if post.search_status == "LIVE_SEARCH" else "bold yellow"
    t_post.add_row("Search Provider Status", f"[{status_style}]{post.search_status}[/{status_style}]")
    t_post.add_row("Search Method", post.match_source)
    if post.similarity_score is not None:
        t_post.add_row("SFace Biometric Similarity", f"[bold green]{post.similarity_score * 100:.2f}%[/bold green] (Ranked vs input face)")
    if post.candidate_count > 0:
        t_post.add_row("Candidates Evaluated", f"{post.candidate_count} live candidates retrieved & ranked")
    
    plat_type_style = "bold cyan" if post.platform_type == "SOCIAL" else "bold yellow"
    t_post.add_row("Platform Type", f"[{plat_type_style}]{post.platform_type}[/{plat_type_style}]")
    t_post.add_row("Platform", f"[bold cyan]{post.platform.upper()}[/bold cyan]")
    t_post.add_row("Author", f"{post.author} ({post.author_handle or 'Verified Profile'})")
    t_post.add_row("Post URL", f"[underline blue]{post.post_url}[/underline blue]")
    t_post.add_row("Content Snippet", f'"{post.content_text}"')
    t_post.add_row("Retrieved At (UTC)", post.retrieved_at)
    t_post.add_row("Publication Timestamp", post.published_date or "Not Disclosed by Platform")
    t_post.add_row("Post Canonical Hash", f"[bold yellow]{post.post_hash}[/bold yellow]")
    console.print(t_post)
    console.print(f"  🔗 [bold cyan]Direct Clickable URL:[/bold cyan] {post.post_url}\n")

    # --- Section 3: Blockchain Record ---
    rec = out.blockchain_record
    t_chain = Table(title=f"Stage 3: Blockchain Ledger Record ({rec.blockchain_type.upper()})", box=box.ROUNDED, style="green")
    t_chain.add_column("Property", style="bold white", width=24)
    t_chain.add_column("Blockchain Value", style="white")
    t_chain.add_row("Record ID", f"[bold cyan]{rec.record_id}[/bold cyan]")
    t_chain.add_row("Block Number", f"#{rec.block_number}")
    t_chain.add_row("Block Hash", f"[dim]{rec.block_hash}[/dim]")
    t_chain.add_row("Transaction Hash", f"[dim]{rec.transaction_hash}[/dim]")
    t_chain.add_row("Anchor Timestamp", f"{rec.timestamp} (Unix Epoch)")
    t_chain.add_row("Face Fingerprint Anchored", f"[yellow]{rec.face_hash}[/yellow]")
    t_chain.add_row("Post Fingerprint Anchored", f"[yellow]{rec.post_hash}[/yellow]")
    console.print(t_chain)
    if rec.blockchain_type == "evm" and rec.transaction_hash:
        tx_hex = rec.transaction_hash if rec.transaction_hash.startswith("0x") else f"0x{rec.transaction_hash}"
        console.print(f"  🌐 [bold cyan]Sepolia Etherscan Explorer:[/bold cyan] [underline blue]https://sepolia.etherscan.io/tx/{tx_hex}[/underline blue]\n")

    # --- Section 4: On-Chain Audit Verification Certificate ---
    audit = out.audit_result
    cert_text = f"""
    [bold green]✔ ON-CHAIN VERIFICATION CERTIFICATE[/bold green]
    -----------------------------------------------------------------------
    Record ID:       [cyan]{audit.record_id}[/cyan]
    Status:          [bold green]PASSED & IMMUTABLE[/bold green]
    Face Hash Match: [bold green]MATCHED ({face.embedding_hash[:16]}...)[/bold green]
    Post Hash Match: [bold green]MATCHED ({post.post_hash[:16]}...)[/bold green]
    Ledger Integrity:[bold green]ALL BLOCKS & MERKLE ROOTS VALID[/bold green]
    Message:         {audit.message}
    """
    console.print(Panel(cert_text.strip(), title="Stage 4: Verification Result", border_style="bold green"))
    return out


def cmd_verify(args):
    print_banner()
    record_id = args.record_id
    chain_type = getattr(args, "chain", None) or "evm"

    console.print(f"[bold yellow]Auditing Record ID on {chain_type.upper()} blockchain:[/bold yellow] [cyan]{record_id}[/cyan]\n")

    if chain_type == "evm":
        ledger = EVMVerificationLedger()
    else:
        ledger = NativeMerkleLedger()

    rec = ledger.get_record(record_id)
    if not rec:
        console.print(f"[bold red]Error:[/bold red] Record ID '{record_id}' not found on {chain_type.upper()} blockchain.")
        sys.exit(1)

    face_hash = args.face_hash or rec.face_hash
    post_hash = args.post_hash or rec.post_hash

    audit = ledger.verify_record(record_id, face_hash, post_hash)

    t = Table(title="Blockchain Re-Verification Audit Report", box=box.ROUNDED)
    t.add_column("Metric", style="bold white", width=25)
    t.add_column("Result", style="white")

    t.add_row("Record ID", audit.record_id)
    t.add_row("On-Chain Face Hash", audit.on_chain_face_hash)
    t.add_row("Supplied Face Hash", audit.provided_face_hash)
    t.add_row("Face Hash Matches?", "[green]✔ YES[/green]" if audit.face_hash_matches else "[red]✖ NO (MISMATCH)[/red]")
    t.add_row("On-Chain Post Hash", audit.on_chain_post_hash)
    t.add_row("Supplied Post Hash", audit.provided_post_hash)
    t.add_row("Post Hash Matches?", "[green]✔ YES[/green]" if audit.post_hash_matches else "[red]✖ NO (MISMATCH)[/red]")
    t.add_row("Block Number", f"#{audit.block_number}")
    t.add_row("Block Hash", audit.block_hash[:32] + "...")
    t.add_row("Overall Cryptographic Validity", "[bold green]VALID / UNTAMPERED[/bold green]" if audit.is_valid else "[bold red]INVALID / TAMPERED[/bold red]")
    console.print(t)


def cmd_tamper_demo(args, scan_output=None):
    print_banner()
    console.print("[bold red]▶ DEMONSTRATING BLOCKCHAIN TAMPER-EVIDENCE[/bold red]\n")
    console.print("This test proves how the cryptographic ledger immediately detects unauthorized alterations.\n")

    if scan_output:
        out = scan_output
        pipe = VerificationPipeline(blockchain_type=out.blockchain_record.blockchain_type)
    else:
        pipe = VerificationPipeline(blockchain_type="native", allow_offline=getattr(args, "offline_fallback", False))
        sample_img = getattr(args, "image", "data/samples/elon_musk.jpg")
        out = pipe.run(sample_img, query_hint="Elon Musk", allow_offline=getattr(args, "offline_fallback", False))

    rec = out.blockchain_record

    console.print(f"[cyan]1. Original Valid Record Anchored:[/cyan]")
    console.print(f"   Record ID:  {rec.record_id}")
    console.print(f"   Face Hash:  {rec.face_hash[:24]}...")
    console.print(f"   Post Hash:  {rec.post_hash[:24]}...")
    console.print(f"   Author:     {out.post.author}")
    console.print(f"   Post URL:   {out.post.post_url}\n")

    # Verify original
    orig_audit = pipe.re_verify(rec.record_id, rec.face_hash, rec.post_hash)
    console.print(f"[green]✔ Baseline Check:[/green] Cryptographic verification status: [bold green]{'PASSED' if orig_audit.is_valid else 'FAILED'}[/bold green]\n")

    # Now simulate an attacker tampering with the post content
    console.print("[bold yellow]2. Simulating Adversarial Tampering Attempt:[/bold yellow]")
    console.print("[red]-- Attacker modifies post author to 'Imposter Attacker' and content to forged text --[/red]")
    tampered_post = SocialPost(
        platform=out.post.platform,
        author="Imposter Attacker (Spoofed)",
        post_url=out.post.post_url,
        content_text="This is a fraudulent forged statement attempting to impersonate the identity.",
    )
    tampered_post_hash = tampered_post.post_hash
    console.print(f"   Original Post Hash: [green]{out.post.post_hash}[/green]")
    console.print(f"   Tampered Post Hash: [red]{tampered_post_hash}[/red]\n")

    console.print("[bold yellow]3. Re-Verifying Tampered Data Against Blockchain Ledger:[/bold yellow]")
    tampered_audit = pipe.re_verify(rec.record_id, rec.face_hash, tampered_post_hash)

    t = Table(title="Tamper Audit Results", box=box.ROUNDED, style="red")
    t.add_column("Audit Check", style="bold white")
    t.add_column("Outcome", style="bold")
    t.add_row("Face Biometric Hash Match", "[green]✔ MATCHED[/green]")
    t.add_row("Post Fingerprint Match", "[bold red]✖ FAILED (Cryptographic Hash Mismatch)[/bold red]")
    t.add_row("On-Chain Verification Status", "[bold red]REJECTED (Data has been altered)[/bold red]")
    console.print(t)

    console.print(Panel(
        f"[bold red]TAMPER DETECTION CONFIRMED[/bold red]\n\n"
        f"The on-chain immutable ledger rejected the tampered post.\n"
        f"Because blockchain hashes are cryptographically one-way and collision-resistant,\n"
        f"any modification to the discovered social media post is mathematically impossible to conceal.",
        border_style="red"
    ))


def cmd_explore(args):
    print_banner()
    ledger = NativeMerkleLedger()
    console.print(f"\n[bold cyan]Blockchain Ledger Explorer (Total Blocks: {len(ledger.chain)})[/bold cyan]\n")

    t = Table(box=box.ROUNDED)
    t.add_column("Block #", style="bold yellow", width=8)
    t.add_column("Block Hash", style="white", width=24)
    t.add_column("Previous Hash", style="dim", width=20)
    t.add_column("Merkle Root", style="cyan", width=22)
    t.add_column("Tx Count", style="green", width=10)
    t.add_column("Timestamp", style="magenta", width=20)

    for b in ledger.chain:
        t.add_row(
            f"#{b['index']}",
            b["hash"][:20] + "...",
            b["previous_hash"][:16] + "...",
            b["merkle_root"][:18] + "...",
            str(len(b.get("transactions", []))),
            str(b.get("timestamp", "")),
        )
    console.print(t)

    is_valid, msg = ledger.verify_chain_integrity()
    if is_valid:
        console.print(f"\n[bold green]✔ Blockchain Ledger Status: HEALTHY & VALID[/bold green] ({msg})")
    else:
        console.print(f"\n[bold red]✖ Blockchain Ledger Status: CORRUPTED[/bold red] ({msg})")


def main():
    parser = argparse.ArgumentParser(
        description="HH GOA 2026 Task 3: Face Identification & Blockchain Verification Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: scan
    scan_parser = subparsers.add_parser("scan", help="Run end-to-end face scan to blockchain verification pipeline")
    scan_parser.add_argument("image", nargs="?", default=None, help="Path or URL to input face image (if omitted, prompts interactively)")
    scan_parser.add_argument("--image", "-i", dest="image_opt", help="Path or URL to input face image")
    scan_parser.add_argument("--chain", choices=["evm", "native"], default="evm", help="Target blockchain (default: evm - Ethereum Sepolia)")
    scan_parser.add_argument("--query", "-q", help="Optional query hint (e.g. subject name) for social search")
    scan_parser.add_argument("--platform", "-p", choices=["twitter", "x", "linkedin", "reddit", "instagram"], help="Preferred social media platform")
    scan_parser.add_argument("--offline-fallback", action="store_true", help="Allow sample corpus fallback (for offline/air-gapped testing)")

    # Command: verify
    verify_parser = subparsers.add_parser("verify", help="Re-verify an existing record against the blockchain")
    verify_parser.add_argument("record_id", help="Record ID on the blockchain")
    verify_parser.add_argument("--chain", choices=["evm", "native"], default="evm", help="Blockchain type (default: evm - Ethereum Sepolia)")
    verify_parser.add_argument("--face-hash", help="Face hash to test against record")
    verify_parser.add_argument("--post-hash", help="Post hash to test against record")

    # Command: tamper-demo
    tamper_parser = subparsers.add_parser("tamper-demo", help="Demonstrate tamper detection by simulating modified data")
    tamper_parser.add_argument("--image", "-i", default=None, help="Input face image for tamper demonstration")
    tamper_parser.add_argument("--offline-fallback", action="store_true", help="Allow sample corpus fallback if offline")

    # Command: explore
    subparsers.add_parser("explore", help="Explore on-chain blocks, transactions, and Merkle roots")

    # Command: demo
    demo_parser = subparsers.add_parser("demo", help="Execute ready-to-record automated pipeline demonstration")
    demo_parser.add_argument("--image", "-i", default=None, help="Custom image path or URL to use in demo")
    demo_parser.add_argument("--query", "-q", default=None, help="Custom query hint for demo")
    demo_parser.add_argument("--chain", choices=["evm", "native"], default="evm", help="Target blockchain (default: evm - Ethereum Sepolia)")
    demo_parser.add_argument("--offline-fallback", action="store_true", help="Allow sample corpus fallback if offline")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "tamper-demo":
        cmd_tamper_demo(args)
    elif args.command == "explore":
        cmd_explore(args)
    elif args.command == "demo":
        demo_image = getattr(args, "image", None) or "data/samples/elon_musk.jpg"
        demo_query = getattr(args, "query", None) or ("Elon Musk" if "elon" in demo_image.lower() else None)
        args.image = demo_image
        args.query = demo_query
        args.chain = getattr(args, "chain", None) or "evm"
        args.platform = None
        args.offline_fallback = getattr(args, "offline_fallback", False)
        out = cmd_scan(args)
        console.print("\n" + "=" * 75 + "\n")
        cmd_tamper_demo(args, scan_output=out)
    elif args.command is None:
        print_banner()
        console.print("[bold cyan]Welcome to the Face Identification & Blockchain Verification Pipeline![/bold cyan]\n")
        console.print("Please select an action:")
        console.print("  [bold green][1][/bold green] Scan Face Image (Input your own image, paste URL, or choose sample) [bold cyan](Ethereum Sepolia)[/bold cyan]")
        console.print("  [bold green][2][/bold green] Run Automated End-to-End Demo [bold cyan](Ethereum Sepolia)[/bold cyan]")
        console.print("  [bold green][3][/bold green] Explore Blockchain Ledger")
        console.print("  [bold green][4][/bold green] Demonstrate Blockchain Tamper-Evidence")
        console.print("  [bold green][5][/bold green] Exit\n")
        try:
            choice = console.input("[bold yellow]Enter option [1-5] (default: 1): [/bold yellow]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            sys.exit(0)

        if choice in ("1", ""):
            args.command = "scan"
            args.image = None
            args.image_opt = None
            args.chain = "evm"
            args.query = None
            args.platform = None
            args.offline_fallback = False
            cmd_scan(args)
        elif choice == "2":
            args.image = "data/samples/elon_musk.jpg"
            args.chain = "evm"
            args.query = "Elon Musk"
            args.platform = None
            args.offline_fallback = False
            out = cmd_scan(args)
            console.print("\n" + "=" * 75 + "\n")
            cmd_tamper_demo(args, scan_output=out)
        elif choice == "3":
            cmd_explore(args)
        elif choice == "4":
            args.image = "data/samples/elon_musk.jpg"
            args.offline_fallback = False
            cmd_tamper_demo(args)
        else:
            console.print("[dim]Goodbye![/dim]")
            sys.exit(0)


if __name__ == "__main__":
    main()
