from agents.structs import FrameData, GameState, GameAction
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import json
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)

def need_to_reset(frames: list[FrameData], latest_frame: FrameData) -> bool:
    return latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]

def need_to_terminate(frames: list[FrameData], latest_frame: FrameData) -> bool:
    return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

def parse_knowledge_sequence(sequence: List[Dict[str, Any]]) -> List[GameAction]:
    """Parse knowledge acquisition sequence and return list of GameActions"""
    parsed_actions = []
    for action_item in sequence:
        try:
            action_name = action_item.get("action", "").upper()
            if hasattr(GameAction, action_name):
                game_action = getattr(GameAction, action_name)
                parsed_actions.append(game_action)
                logger.info(f"[OK] Parsed action: {action_name} (value: {game_action.value})")
            else:
                logger.warning(f"[WARN] Unknown action: {action_name}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to parse action item {action_item}: {e}")
    
    return parsed_actions

def write_knowledge_response(sequence: List[Dict[str, Any]], frame_data: FrameData, parsed_actions: List[GameAction]) -> None:
    """Write knowledge acquisition response to temp folder"""
    
    # Create temp directory
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    # Custom serializer for FrameData
    def serialize_frame_data(frame_data: FrameData) -> dict:
        """Convert FrameData to JSON-serializable dictionary"""
        return {
            "game_id": frame_data.game_id,
            "frame": frame_data.frame,  # This should already be a list of lists of ints
            "state": frame_data.state.value,  # Convert enum to its value
            "score": frame_data.score,
            "action_input": {
                "id": frame_data.action_input.id.value,  # Convert GameAction enum to its integer value
                "data": frame_data.action_input.data,    # This should already be a dict
                "reasoning": frame_data.action_input.reasoning
            },
            "guid": frame_data.guid,
            "full_reset": frame_data.full_reset
        }
    
    # Prepare response payload
    response_payload = {
        "timestamp": timestamp,
        "frame_data": serialize_frame_data(frame_data),
        "knowledge_acquisition_sequence": sequence,
        "parsed_actions": [
            {
                "action_name": action.name,
                "action_value": action.value,
                "expectation": next((item.get("expectation", "") for item in sequence if item.get("action", "").upper() == action.name), ""),
                "coordinates": next((item.get("coordinates", None) for item in sequence if item.get("action", "").upper() == action.name), None)
            }
            for action in parsed_actions
        ]
    }
    
    # Write JSON payload file
    try:
        payload_file = temp_dir / f"knowledge_response_{timestamp}.json"
        with open(payload_file, 'w', encoding='utf-8') as f:
            json.dump(response_payload, f, indent=2, ensure_ascii=False)
        logger.info(f"[SUCCESS] JSON payload saved: {payload_file}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to write JSON payload: {e}")
    
    # Write summary file
    try:
        summary_file = temp_dir / f"knowledge_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Knowledge Acquisition Summary\n")
            f.write(f"============================\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Game ID: {frame_data.game_id}\n")
            f.write(f"Frame State: {frame_data.state}\n")
            f.write(f"Frame Score: {frame_data.score}\n")
            f.write(f"Action Input ID: {frame_data.action_input.id.name} (value: {frame_data.action_input.id.value})\n")
            f.write(f"Full Reset: {frame_data.full_reset}\n\n")
            f.write(f"Total Actions Generated: {len(sequence)}\n")
            f.write(f"Successfully Parsed Actions: {len(parsed_actions)}\n\n")
            f.write("Parsed Actions:\n")
            for i, action in enumerate(parsed_actions, 1):
                f.write(f"  {i}. {action.name} (value: {action.value})\n")
            f.write(f"\nJSON payload: {payload_file.name}\n")
        logger.info(f"[SUCCESS] Summary saved: {summary_file}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to write summary: {e}")
    
    logger.info(f"[INFO] Knowledge response files saved to: {temp_dir}")

def frame_to_png(frame_data: FrameData, output_path: Path, title: str = "", action_info: str = "") -> bool:
    """Convert FrameData to PNG image with color mapping and action information"""
    try:
        if not frame_data.frame or len(frame_data.frame) == 0:
            logger.warning(f"Empty frame data, cannot create PNG: {output_path}")
            return False
        
        # Convert frame to numpy array
        frame_array = np.array(frame_data.frame)
        
        # Handle different frame structures
        if len(frame_array.shape) == 3:
            # If it's 3D with shape (64, 64, 1), squeeze the last dimension
            if frame_array.shape[2] == 1:
                frame_array = np.squeeze(frame_array, axis=2)
            # If channels first (e.g., [1, 64, 64])
            elif frame_array.shape[0] == 1:
                frame_array = np.squeeze(frame_array, axis=0)
            # If it's multi-channel, take first channel
            elif frame_array.shape[2] > 1:
                frame_array = frame_array[:, :, 0]
            # If shape is like [channels, height, width] with channels > 1
            elif frame_array.shape[0] <= 10 and frame_array.shape[0] > 1:
                frame_array = frame_array[0, :, :]
        
        # Ensure 2D array
        if len(frame_array.shape) != 2:
            logger.error(f"Unexpected frame shape after processing: {frame_array.shape}, original: {np.array(frame_data.frame).shape}")
            return False
        
        logger.info(f"[PNG] Processing frame shape: {frame_array.shape}, min: {np.min(frame_array)}, max: {np.max(frame_array)}")
        
        # Create color map for values 0-15 (extended for ARC puzzles)
        colors = [
            '#000000',  # 0: Black
            '#0074D9',  # 1: Blue
            '#FF4136',  # 2: Red  
            '#2ECC40',  # 3: Green
            '#FFDC00',  # 4: Yellow
            '#AAAAAA',  # 5: Gray
            '#F012BE',  # 6: Magenta/Pink
            '#FF851B',  # 7: Orange
            '#7FDBFF',  # 8: Aqua/Light Blue
            '#870C25',  # 9: Maroon
            '#FFFFFF',  # 10: White
            '#85144b',  # 11: Navy
            '#3D9970',  # 12: Teal
            '#B10DC9',  # 13: Purple
            '#01FF70',  # 14: Lime
            '#FFEAA7'   # 15: Light Yellow
        ]
        
        # Extend colors if needed for values beyond 15
        max_val = int(np.max(frame_array))
        while len(colors) <= max_val:
            colors.append(f'#{np.random.randint(0, 256):02x}{np.random.randint(0, 256):02x}{np.random.randint(0, 256):02x}')
        
        cmap = mcolors.ListedColormap(colors[:max_val + 1])
        
        # Create figure with more space for action info
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        
        # Display the frame
        im = ax.imshow(frame_array, cmap=cmap, vmin=0, vmax=max_val, interpolation='nearest')
        
        # Add grid lines for better visibility
        height, width = frame_array.shape
        for i in range(height + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.8)
        for j in range(width + 1):
            ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.8)
        
        # Create comprehensive title with action information
        title_parts = []
        if title:
            title_parts.append(title)
        if action_info:
            title_parts.append(f"Action: {action_info}")
        
        main_title = " | ".join(title_parts)
        
        ax.set_title(f"{main_title}\nState: {frame_data.state.value} | Score: {frame_data.score} | Grid: {width}x{height}", 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add action information as text on the plot if provided
        if action_info:
            ax.text(0.02, 0.98, f"ACTION: {action_info}", 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Set axis labels
        ax.set_xlabel(f"Columns (0-{width-1})", fontsize=10)
        ax.set_ylabel(f"Rows (0-{height-1})", fontsize=10)
        
        # Show some tick marks for reference (every 8 cells)
        ax.set_xticks(range(0, width, 8))
        ax.set_yticks(range(0, height, 8))
        
        # Add colorbar if there are multiple values
        if max_val > 0:
            cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=30)
            cbar.set_label('Cell Values', rotation=270, labelpad=15, fontsize=10)
            
            # Add value labels to colorbar
            if max_val <= 15:  # Only for reasonable number of values
                cbar.set_ticks(range(max_val + 1))
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"[SUCCESS] Frame PNG saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to create PNG for frame: {e}")
        return False

def write_action_analysis(action_name: str, analysis_result: Dict[str, Dict[str, Any]], before_frame: FrameData, after_frame: FrameData) -> None:
    """Write action analysis to temp folder with PNG images"""
    
    # Create temp directory
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    # Create PNG images for before and after frames
    before_png_path = temp_dir / f"action_analysis_{action_name}_{timestamp}_before.png"
    after_png_path = temp_dir / f"action_analysis_{action_name}_{timestamp}_after.png"
    
    frame_to_png(before_frame, before_png_path, f"BEFORE - {action_name}")
    frame_to_png(after_frame, after_png_path, f"AFTER - {action_name}")
    
    # Prepare analysis payload with enhanced expectation information
    analysis_payload = {
        "timestamp": timestamp,
        "action_name": action_name,
        "analysis": analysis_result,
        "expectation_details": {
            "expectation_matched": analysis_result.get(action_name, {}).get("expectation_matched", "NO_EXPECTATION"),
            "expectation_explanation": analysis_result.get(action_name, {}).get("expectation_explanation", "No explanation provided"),
            "reasoning": analysis_result.get(action_name, {}).get("reasoning", "No reasoning"),
            "confidence": analysis_result.get(action_name, {}).get("confidence", "UNKNOWN")
        },
        "before_frame": {
            "frame": before_frame.frame,
            "state": before_frame.state.value,
            "score": before_frame.score,
            "png_file": before_png_path.name
        },
        "after_frame": {
            "frame": after_frame.frame,
            "state": after_frame.state.value,
            "score": after_frame.score,
            "png_file": after_png_path.name
        }
    }
    
    # Write analysis file
    try:
        analysis_file = temp_dir / f"action_analysis_{action_name}_{timestamp}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_payload, f, indent=2, ensure_ascii=False)
        logger.info(f"[SUCCESS] Action analysis saved: {analysis_file}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to write action analysis: {e}")

def write_sequence_analysis(sequence: List[Dict[str, Any]], frames_sequence: List[FrameData], analysis_results: List[Dict[str, Any]]) -> None:
    """Write comprehensive sequence analysis to temp folder with PNG images"""
    
    # Create temp directory
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    # Create subdirectory for this sequence's images
    images_dir = temp_dir / f"sequence_images_{timestamp}"
    images_dir.mkdir(exist_ok=True)
    
    # Custom serializer for FrameData with action information
    def serialize_frame_data(frame_data: FrameData, step: int = 0, prefix: str = "", action_name: str = "", action_analysis: Dict = None) -> dict:
        """Convert FrameData to JSON-serializable dictionary with PNG path"""
        
        # Create descriptive filename
        if action_name:
            png_filename = f"step_{step:02d}_{prefix}_{action_name}.png"
        else:
            png_filename = f"step_{step:02d}_{prefix}.png"
        png_path = images_dir / png_filename
        
        # Create title and action info for PNG
        if prefix == "before" and action_name:
            title = f"Step {step} - BEFORE {action_name}"
            action_info = action_name
        elif prefix == "after" and action_name:
            title = f"Step {step} - AFTER {action_name}"
            action_info = action_name
            # Add analysis reasoning if available
            if action_analysis and action_name in action_analysis:
                reasoning = action_analysis[action_name].get("reasoning", "")
                confidence = action_analysis[action_name].get("confidence", "UNKNOWN")
                expectation_matched = action_analysis[action_name].get("expectation_matched", "NO_EXPECTATION")
                friendly_name = action_analysis[action_name].get("name", action_name)
                action_info = f"{action_name} ({friendly_name}) - {confidence}"
                if expectation_matched != "NO_EXPECTATION":
                    action_info += f" | Expected: {expectation_matched}"
        else:
            title = f"Step {step} - {prefix}"
            action_info = ""
        
        # Create PNG image with action information
        frame_to_png(frame_data, png_path, title, action_info)
        
        return {
            "game_id": frame_data.game_id,
            "frame": frame_data.frame,
            "state": frame_data.state.value,
            "score": frame_data.score,
            "action_input": {
                "id": frame_data.action_input.id.value,
                "data": frame_data.action_input.data,
                "reasoning": frame_data.action_input.reasoning
            },
            "guid": frame_data.guid,
            "full_reset": frame_data.full_reset,
            "png_file": png_filename,
            "png_path": str(png_path.relative_to(temp_dir))
        }
    
    # Prepare comprehensive analysis payload
    sequence_analysis = {
        "timestamp": timestamp,
        "sequence_length": len(sequence),
        "images_directory": str(images_dir.relative_to(temp_dir)),
        "analysis_summary": {
            "total_actions": len(sequence),
            "successful_analyses": len([r for r in analysis_results if r]),
            "failed_analyses": len([r for r in analysis_results if not r])
        },
        "detailed_analysis": []
    }
    
    # Add detailed analysis for each step
    for i, (action_item, analysis_result) in enumerate(zip(sequence, analysis_results)):
        action_name = action_item.get("action", "").upper()
        expectation = action_item.get("expectation", "")
        coordinates = action_item.get("coordinates", None)
        
        step_analysis = {
            "step": i + 1,
            "action_name": action_name,
            "expectation": expectation,
            "coordinates": coordinates,
            "analysis": analysis_result if analysis_result else {"error": "Analysis failed"},
        }
        
        # Add before/after frames with PNG images if available
        if i < len(frames_sequence) - 1:
            step_analysis["before_frame"] = serialize_frame_data(
                frames_sequence[i], i + 1, "before", action_name, analysis_result
            )
            step_analysis["after_frame"] = serialize_frame_data(
                frames_sequence[i + 1], i + 1, "after", action_name, analysis_result
            )
        elif i == 0 and len(frames_sequence) > 0:
            # For the first step, we might only have the initial frame
            step_analysis["initial_frame"] = serialize_frame_data(
                frames_sequence[0], 0, "initial", "", None
            )
        
        sequence_analysis["detailed_analysis"].append(step_analysis)
    
    # Write comprehensive analysis file
    try:
        analysis_file = temp_dir / f"sequence_analysis_{timestamp}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(sequence_analysis, f, indent=2, ensure_ascii=False)
        logger.info(f"[SUCCESS] Sequence analysis saved: {analysis_file}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to write sequence analysis: {e}")
    
    # Write summary file with clearer action descriptions
    try:
        summary_file = temp_dir / f"sequence_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Action Sequence Analysis Summary\n")
            f.write(f"==============================\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Actions: {len(sequence)}\n")
            f.write(f"Successful Analyses: {len([r for r in analysis_results if r])}\n")
            f.write(f"Failed Analyses: {len([r for r in analysis_results if not r])}\n")
            f.write(f"Images Directory: {images_dir.name}\n\n")
            
            f.write("Action Summary:\n")
            f.write("================\n")
            for i, (action_item, analysis_result) in enumerate(zip(sequence, analysis_results)):
                action_name = action_item.get("action", "").upper()
                expectation = action_item.get("expectation", "")
                
                if analysis_result and action_name in analysis_result:
                    reasoning = analysis_result[action_name].get("reasoning", "No reasoning")
                    confidence = analysis_result[action_name].get("confidence", "UNKNOWN")
                    expectation_matched = analysis_result[action_name].get("expectation_matched", "NO_EXPECTATION")
                    expectation_explanation = analysis_result[action_name].get("expectation_explanation", "No explanation provided")
                    action_name_friendly = analysis_result[action_name].get("name", action_name)
                    
                    f.write(f"\n{i+1}. {action_name} - '{action_name_friendly}':\n")
                    f.write(f"   Reasoning: {reasoning}\n")
                    f.write(f"   Confidence: {confidence}\n")
                    f.write(f"   Images:\n")
                    f.write(f"     - Before: step_{i+1:02d}_before_{action_name}.png\n")
                    f.write(f"     - After:  step_{i+1:02d}_after_{action_name}.png\n")
                    if expectation:
                        f.write(f"   Expected: {expectation[:100]}...\n")
                        f.write(f"   Expectation Match: {expectation_matched}\n")
                        f.write(f"   Match Explanation: {expectation_explanation}\n")
                else:
                    f.write(f"\n{i+1}. {action_name}: Analysis failed\n")
                    f.write(f"   Images:\n")
                    f.write(f"     - Before: step_{i+1:02d}_before_{action_name}.png\n")
                    f.write(f"     - After:  step_{i+1:02d}_after_{action_name}.png\n")
            
            f.write(f"\nFiles Generated:\n")
            f.write(f"- Detailed analysis: {analysis_file.name}\n")
            f.write(f"- PNG Images: {images_dir.name}/\n")
            f.write(f"- This summary: {summary_file.name}\n")
        logger.info(f"[SUCCESS] Sequence summary saved: {summary_file}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to write sequence summary: {e}")