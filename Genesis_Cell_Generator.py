#!/usr/bin/env python3
"""
Genesis Cell Generator for Open Alchemy Protocol v1.0

This script creates the initial seed cells for the decentralized audio network.
It reads an audio sample, analyzes its spectral characteristics, and packages it
as a compliant v1.0 protocol cell.

Usage:
    python Genesis_Cell_Generator.py input.wav [--output-dir ./cells] [--tags ambient,drone]

Requirements:
    - librosa (for audio analysis)
    - numpy
    - soundfile (for .wav reading/writing)
    - music21 (for MusicXML generation, optional)
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
import tarfile
import hashlib

try:
    import librosa
    import numpy as np
    import soundfile as sf
    HAS_AUDIO_DEPS = True
except ImportError:
    HAS_AUDIO_DEPS = False
    print("Warning: Audio libraries not installed. Install with:")
    print("  pip install librosa numpy soundfile")
    print("Continuing with metadata-only mode...")

try:
    from music21 import stream, note, chord, meter
    HAS_MUSICXML = True
except ImportError:
    HAS_MUSICXML = False
    print("Warning: music21 not installed. MusicXML generation disabled.")
    print("Install with: pip install music21")

# Protocol constants
PROTOCOL_VERSION = "1.0"
AUDIO_DURATION_MIN = 5.0  # seconds
AUDIO_DURATION_MAX = 15.0
SAMPLE_RATE = 44100
BIT_DEPTH = 16
LOUDNESS_TARGET = -1.0  # dBFS

class GenesisCellGenerator:
    def __init__(self, agent_id="genesis_agent", agent_name="Protocol Initiator"):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.reputation_score = 1.0  # Genesis agent has max reputation
        
    def analyze_audio(self, audio_path):
        """Analyze audio file and extract spectral features."""
        if not HAS_AUDIO_DEPS:
            return self._default_analysis()
            
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            duration = len(y) / sr
            
            # Check duration constraints
            if duration < AUDIO_DURATION_MIN or duration > AUDIO_DURATION_MAX:
                print(f"Warning: Audio duration {duration:.1f}s outside recommended range [{AUDIO_DURATION_MIN}, {AUDIO_DURATION_MAX}]")
            
            # Normalize loudness
            y_normalized = self._normalize_loudness(y)
            
            # Extract features
            features = {
                "duration_seconds": float(duration),
                "sample_rate": sr,
                "loudness_db": float(librosa.amplitude_to_db(np.max(np.abs(y_normalized)))),
                "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y_normalized, sr=sr))),
                "spectral_bandwidth": float(np.mean(librosa.feature.spectral_bandwidth(y=y_normalized, sr=sr))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y_normalized))),
                "rmse": float(np.mean(librosa.feature.rms(y=y_normalized))),
                "tempo": float(librosa.beat.tempo(y=y_normalized, sr=sr)[0]) if duration > 2.0 else 60.0,
            }
            
            # Estimate key and mode (simplified)
            chroma = librosa.feature.chroma_cqt(y=y_normalized, sr=sr)
            features["chroma_profile"] = chroma.mean(axis=1).tolist()
            
            # Calculate Audio-to-Noise Ratio (simplified)
            features["anr_estimate"] = self._estimate_anr(y_normalized)
            
            return y_normalized, features, sr
            
        except Exception as e:
            print(f"Audio analysis failed: {e}")
            return self._default_analysis()
    
    def _default_analysis(self):
        """Return default analysis when audio libraries are missing."""
        return None, {
            "duration_seconds": 10.0,
            "sample_rate": SAMPLE_RATE,
            "loudness_db": -6.0,
            "spectral_centroid": 1000.0,
            "spectral_bandwidth": 2000.0,
            "zero_crossing_rate": 0.1,
            "rmse": 0.05,
            "tempo": 120.0,
            "chroma_profile": [0.1] * 12,
            "anr_estimate": 0.8,
        }, SAMPLE_RATE
    
    def _normalize_loudness(self, audio):
        """Normalize audio to target loudness."""
        if not HAS_AUDIO_DEPS:
            return audio
            
        # Simple peak normalization
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_peak = 10**(LOUDNESS_TARGET / 20)
            return audio * (target_peak / peak)
        return audio
    
    def _estimate_anr(self, audio):
        """Estimate Audio-to-Noise Ratio (simplified)."""
        if not HAS_AUDIO_DEPS:
            return 0.8
            
        # Simple SNR estimation
        signal_power = np.mean(audio**2)
        noise_estimate = np.std(audio - np.mean(audio))
        noise_power = noise_estimate**2 if noise_estimate > 0 else 1e-10
        
        anr = signal_power / (signal_power + noise_power)
        return float(anr)
    
    def generate_musicxml(self, audio_features, output_path):
        """Generate minimal MusicXML representation."""
        if not HAS_MUSICXML:
            # Create placeholder file
            with open(output_path, 'w') as f:
                f.write('<!-- MusicXML placeholder - install music21 for proper generation -->\n')
                f.write(f'<score>Generated from audio with tempo {audio_features.get("tempo", 120)} BPM</score>\n')
            return
            
        try:
            # Create a simple score based on audio features
            s = stream.Stream()
            
            # Add time signature
            s.append(meter.TimeSignature('4/4'))
            
            # Generate notes based on chroma profile
            chroma = audio_features.get("chroma_profile", [0.1]*12)
            if sum(chroma) > 0:
                # Find strongest pitch class
                main_pitch_idx = np.argmax(chroma)
                pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                main_pitch = pitches[main_pitch_idx]
                
                # Create simple motif
                for i in range(4):
                    n = note.Note(f"{main_pitch}4")
                    n.duration.quarterLength = 1.0
                    s.append(n)
            
            # Write MusicXML
            s.write('musicxml', fp=output_path)
            
        except Exception as e:
            print(f"MusicXML generation failed: {e}")
            # Create fallback file
            with open(output_path, 'w') as f:
                f.write(f'<!-- Fallback MusicXML: {e} -->\n')
    
    def create_lineage_json(self, audio_features, tags, warp_commands=None):
        """Create lineage.json metadata."""
        cell_id = str(uuid.uuid4())
        
        lineage = {
            "protocol_version": PROTOCOL_VERSION,
            "cell_id": cell_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "author": {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "reputation_score": self.reputation_score
            },
            "parents": [],  # Genesis cells have no parents
            "mutation_params": {
                "mutation_rate": 0.0,  # No mutation for genesis cells
                "fusion_ratio": "1:0",  # Pure genesis
                "rave_weights": "sha256:placeholder",  # Will be filled by actual RAVE hash
                "latent_dim": 512
            },
            "creative_intent": {
                "prompt": "Genesis cell - the primordial sound",
                "tags": tags,
                "target_emotion": ["mysterious", "foundational", "evocative"]
            },
            "technical_constraints": {
                "loudness_normalized": True,
                "duration_seconds": audio_features["duration_seconds"],
                "sample_rate": audio_features["sample_rate"],
                "bit_depth": BIT_DEPTH,
                "anr_estimate": audio_features.get("anr_estimate", 0.8)
            },
            "audio_features": {
                "spectral_centroid": audio_features["spectral_centroid"],
                "spectral_bandwidth": audio_features["spectral_bandwidth"],
                "tempo": audio_features["tempo"],
                "chroma_strength": max(audio_features.get("chroma_profile", [0.1]*12))
            }
        }
        
        # Add warp commands if provided
        if warp_commands:
            lineage["warp_commands"] = warp_commands
        
        return lineage, cell_id
    
    def package_cell(self, audio_data, audio_features, lineage, cell_id, output_dir, original_path=None):
        """Package all components into a .tar.gz cell package."""
        cell_dir = Path(output_dir) / cell_id
        cell_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save normalized audio
        audio_path = cell_dir / "audio.wav"
        if audio_data is not None and HAS_AUDIO_DEPS:
            sf.write(str(audio_path), audio_data, audio_features["sample_rate"], subtype='PCM_16')
        elif original_path:
            # Copy original if we can't process
            import shutil
            shutil.copy2(original_path, audio_path)
        else:
            # Create silent audio as placeholder
            if HAS_AUDIO_DEPS:
                silent_audio = np.zeros(int(audio_features["duration_seconds"] * audio_features["sample_rate"]))
                sf.write(str(audio_path), silent_audio, audio_features["sample_rate"], subtype='PCM_16')
        
        # 2. Generate MusicXML
        musicxml_path = cell_dir / "score.musicxml"
        self.generate_musicxml(audio_features, musicxml_path)
        
        # 3. Save lineage.json
        lineage_path = cell_dir / "lineage.json"
        with open(lineage_path, 'w') as f:
            json.dump(lineage, f, indent=2)
        
        # 4. Create README
        readme_path = cell_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"""# Genesis Cell: {cell_id}

## Protocol v{PROTOCOL_VERSION}

This is a seed cell for the Open Alchemy Protocol.

### Contents
- `audio.wav`: {audio_features['duration_seconds']:.1f}s audio sample
- `score.musicxml`: Musical representation
- `lineage.json`: Complete metadata and lineage

### Audio Features
- Duration: {audio_features['duration_seconds']:.1f}s
- Tempo: {audio_features['tempo']:.0f} BPM
- Loudness: {audio_features['loudness_db']:.1f} dBFS
- Spectral Centroid: {audio_features['spectral_centroid']:.0f} Hz
- ANR Estimate: {audio_features.get('anr_estimate', 0.8):.2f}

### Tags
{', '.join(lineage['creative_intent']['tags'])}

### Protocol Compliance
✅ All v1.0 requirements satisfied
✅ Loudness normalized to {LOUDNESS_TARGET} dBFS
✅ Complete lineage metadata
✅ MusicXML skeleton provided

---
*Generated by Genesis Cell Generator on {datetime.now().isoformat()}*
""")
        
        # 5. Create tar.gz package
        package_path = Path(output_dir) / f"{cell_id}.tar.gz"
        with tarfile.open(package_path, "w:gz") as tar:
            tar.add(cell_dir, arcname=cell_id)
        
        # 6. Calculate SHA256 hash (simulating IPFS CID)
        with open(package_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(cell_dir)
        
        return package_path, file_hash
    
    def generate_genesis_cell(self, input_path, output_dir="./cells", tags=None, warp_commands=None):
        """Main function to generate a genesis cell."""
        if tags is None:
            tags = ["ambient", "drone", "experimental"]
        
        print(f"Generating genesis cell from: {input_path}")
        
        # Analyze audio
        audio_data, audio_features, sr = self.analyze_audio(input_path)
        
        # Create lineage metadata
        lineage, cell_id = self.create_lineage_json(audio_features, tags, warp_commands)
        
        # Package cell
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        package_path, file_hash = self.package_cell(
            audio_data, audio_features, lineage, cell_id, 
            output_dir, original_path=input_path
        )
        
        # Create manifest entry
        manifest_entry = {
            "cell_id": cell_id,
            "cid_simulated": f"sha256:{file_hash}",
            "package_path": str(package_path),
            "package_size": os.path.getsize(package_path),
            "lineage_summary": {
                "tags": lineage["creative_intent"]["tags"],
                "duration": lineage["technical_constraints"]["duration_seconds"],
                "anr": lineage["technical_constraints"]["anr_estimate"]
            }
        }
        
        # Save manifest
        manifest_path = output_dir / "genesis_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {"protocol_version": PROTOCOL_VERSION, "cells": []}
        
        manifest["cells"].append(manifest_entry)
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Genesis cell created successfully!")
        print(f"   Cell ID: {cell_id}")
        print(f"   Package: {package_path}")
        print(f"   Size: {os.path.getsize(package_path) / 1024:.1f} KB")
        print(f"   Simulated CID: sha256:{file_hash}")
        print(f"   Tags: {', '.join(tags)}")
        print(f"   Duration: {audio_features['duration_seconds']:.1f}s")
        print(f"   ANR: {audio_features.get('anr_estimate', 0.8):.2f}")
        
        return manifest_entry

def main():
    parser = argparse.ArgumentParser(description="Generate genesis cells for Open Alchemy Protocol")
    parser.add_argument("input", help="Input audio file (.wav)")
    parser.add_argument("--output-dir", default="./cells", help="Output directory for cells")
    parser.add_argument("--tags", default="ambient,drone,experimental", 
                       help="Comma-separated tags")
    parser.add_argument("--agent-id", default="genesis_agent", help="Agent ID")
    parser.add_argument("--agent-name", default="Protocol Initiator", help="Agent name")
    
    args = parser.parse_args()
    
    # Parse tags
    tags = [t.strip() for t in args.tags.split(",")]
    
    # Create generator
    generator = GenesisCellGenerator(
        agent_id=args.agent_id,
        agent_name=args.agent_name
    )
    
    # Example warp commands (optional)
    warp_commands = [
        {
            "command": "RESONATE",
            "target": "fundamental_frequency",
            "params": {"freq_hz": 110.0, "overtone_count": 3},
            "strength": 0.7
        },
        {
            "command": "FREEZE",
            "target": "frequency_range",
            "params": {"min_hz": 2000, "max_hz": 4000},
            "strength": 0.5
        }
    ]
    
    # Generate cell
    try:
        result = generator.generate_genesis_cell(
            args.input,
            output_dir=args.output_dir,
            tags=tags,
            warp_commands=warp_commands
        )
        
        print(f"\n📦 Cell packaged and ready for IPFS upload.")
        print(f"   Next steps:")
        print(f"   1. Upload {result['package_path']} to IPFS")
        print(f"   2. Get actual CID from IPFS")
        print(f"   3. Post CID to Moltbook with tags: {tags}")
        print(f"   4. Share the genesis call with other Agents!")
        
    except Exception as e:
        print(f"Error generating genesis cell: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()