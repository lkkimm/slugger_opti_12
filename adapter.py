# adapter.py — SLUGGER API Request Adapter
# -*- coding: utf-8 -*-
"""
Module for fetching real baseball data through the SLUGGER API.

"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

# Load .env file
load_dotenv()

log = logging.getLogger(__name__)

# API Base Configuration
BASE_URL = "https://1ywv9dczq5.execute-api.us-east-2.amazonaws.com/ALPBAPI"
API_KEY = os.getenv("API_KEY")

# Local fallback spray data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "spray"


def _load_spray_from_local_file(player_id: str, pitcher_hand: Optional[str] = None) -> List[Dict]:
    """
    Load spray data from a local JSON file (fallback if API fails).
    
    Args:
        player_id: Batter ID
        pitcher_hand: Pitcher throwing hand filter (R/L, optional)
    
    Returns:
        List[Dict]: Batted ball data, or empty list if no file or invalid.
    """
    spray_file = DATA_DIR / f"{player_id}.json"
    
    if not spray_file.exists():
        log.debug(f"Local file does not exist: {spray_file}")
        return []
    
    try:
        with open(spray_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            pitches_data = data.get("data", [])
        elif isinstance(data, list):
            pitches_data = data
        else:
            log.warning(f"Unexpected JSON format: {spray_file}")
            return []
        
        if not pitches_data:
            log.debug(f"Local file contains no data: {spray_file}")
            return []
        
        log.info(f"Loaded {len(pitches_data)} entries from local file")
        
        filtered_pitches = [
            p for p in pitches_data
            if any([
                p.get("hit_trajectory_xc2") is not None,
                p.get("hit_trajectory_xc1") is not None,
                p.get("hit_trajectory_xc0") is not None,
                p.get("direction") is not None,
                p.get("exit_speed") is not None
            ])
        ]
        
        log.info(f"Valid batted-ball entries: {len(filtered_pitches)}/{len(pitches_data)}")
        
        if pitcher_hand and filtered_pitches:
            pitcher_hand_upper = pitcher_hand.replace("HP", "").upper()
            original_count = len(filtered_pitches)
            
            def normalize_pitcher_throws(value):
                if not value:
                    return None
                v = str(value).upper().strip()
                if v.startswith("RIGHT") or v == "R":
                    return "R"
                if v.startswith("LEFT") or v == "L":
                    return "L"
                return None
            
            filtered_pitches = [
                p for p in filtered_pitches
                if normalize_pitcher_throws(p.get("pitcher_throws")) == pitcher_hand_upper
            ]
            
            log.info(
                f"Pitcher Hand Filter ({pitcher_hand_upper}): "
                f"{len(filtered_pitches)}/{original_count}"
            )
        
        return filtered_pitches
        
    except Exception as e:
        log.error(f"Failed to load local file: {spray_file}, Error: {e}")
        return []


if not API_KEY:
    log.warning("API_KEY not found in .env file. API calls will fail.")

HEADERS = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}


def fetch_ballparks(ballpark_name: Optional[str] = None,
                    city: Optional[str] = None,
                    state: Optional[str] = None,
                    limit: int = 50,
                    page: int = 1,
                    order: str = "ASC") -> List[Dict]:
    """
    Fetch list of ballparks.
    """
    url = f"{BASE_URL}/ballparks"
    params = {"limit": limit, "page": page, "order": order}
    
    if ballpark_name:
        params["ballpark_name"] = ballpark_name
    if city:
        params["city"] = city
    if state:
        params["state"] = state
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data.get("data", [])
        
        log.error(f"API error: {data.get('message')}")
        return []
    
    except Exception as e:
        log.error(f"Failed to fetch ballparks: {e}")
        return []


def fetch_games(ballpark_name: Optional[str] = None,
                team_name: Optional[str] = None,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                limit: int = 50,
                page: int = 1,
                order: str = "DESC") -> List[Dict]:
    """
    Fetch list of games.
    """
    url = f"{BASE_URL}/games"
    params = {"limit": limit, "page": page, "order": order}
    
    if ballpark_name:
        params["ballpark_name"] = ballpark_name
    if team_name:
        params["team_name"] = team_name
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data.get("data", [])
        
        log.error(f"API error: {data.get('message')}")
        return []
    
    except Exception as e:
        log.error(f"Failed to fetch games: {e}")
        return []


def fetch_player_spray(player_id: str,
                       pitcher_hand: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       limit: int = 5000) -> List[Dict]:
    """
    Fetch spray chart data for a specific batter using the /pitches endpoint.
    """
    url = f"{BASE_URL}/pitches"
    params = {"batter_id": player_id, "limit": min(limit, 1000)}
    
    if start_date:
        params["date_range_start"] = start_date
    if end_date:
        params["date_range_end"] = end_date
    
    log.info(f"Pitches API request: {url}")
    
    max_retries = 2
    retry_count = 0
    data = None
    
    while retry_count <= max_retries:
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            log.info(f"HTTP Status Code: {response.status_code}")
            
            if response.status_code == 502:
                retry_count += 1
                if retry_count <= max_retries:
                    import time
                    time.sleep(retry_count * 2)
                    continue
                raise requests.exceptions.HTTPError("502 after retries")
            
            response.raise_for_status()
            data = response.json()
            break
        
        except requests.exceptions.RequestException as e:
            retry_count += 1
            if retry_count <= max_retries:
                import time
                time.sleep(retry_count * 2)
                continue
            
            log.error(f"API request failed: {str(e)}")
            fallback = _load_spray_from_local_file(player_id, pitcher_hand)
            return fallback
    
    if not data:
        fallback = _load_spray_from_local_file(player_id, pitcher_hand)
        return fallback
    
    if data.get("success"):
        pitches_data = data.get("data", [])
        
        filtered_pitches = [
            p for p in pitches_data
            if any([
                p.get("hit_trajectory_xc2") is not None,
                p.get("hit_trajectory_xc1") is not None,
                p.get("hit_trajectory_xc0") is not None,
                p.get("direction") is not None,
                p.get("exit_speed") is not None
            ])
        ]
        
        if pitcher_hand:
            pitcher_hand_upper = pitcher_hand.replace("HP", "").upper()
            
            def normalize(value):
                if not value:
                    return None
                v = str(value).upper().strip()
                if v.startswith("RIGHT") or v == "R":
                    return "R"
                if v.startswith("LEFT") or v == "L":
                    return "L"
                return None
            
            filtered_pitches = [
                p for p in filtered_pitches
                if normalize(p.get("pitcher_throws")) == pitcher_hand_upper
            ]
        
        return filtered_pitches
    
    else:
        fallback = _load_spray_from_local_file(player_id, pitcher_hand)
        return fallback


def fetch_players(team_name: Optional[str] = None,
                  handedness: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  limit: int = 5000) -> List[Dict]:
    """
    Fetch player list.
    """
    url = f"{BASE_URL}/players"
    params = {"limit": min(limit, 1000)}
    
    if team_name:
        params["team_name"] = team_name
    
    if handedness:
        h = handedness.upper()
        if h in ["LEFT", "L"]:
            params["player_batting_handedness"] = "Left"
        elif h in ["RIGHT", "R"]:
            params["player_batting_handedness"] = "Right"
        elif h in ["SWITCH", "S"]:
            params["player_batting_handedness"] = "Switch"
        else:
            params["player_batting_handedness"] = handedness
    
    if not API_KEY:
        log.error("API_KEY missing.")
        return []
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data.get("data", [])
        
        log.error(f"API Error: {data.get('message')}")
        return []
    
    except Exception as e:
        log.error(f"API request failed: {e}")
        return []


def fetch_batted_balls(player_ids: Optional[List[str]] = None,
                       handedness: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       limit: int = 5000) -> List[Dict]:
    """
    Fetch bulk batted-ball data.
    """
    url = f"{BASE_URL}/atbats"
    params = {"limit": limit}
    
    if player_ids:
        params["player_ids"] = ",".join(player_ids)
    if handedness:
        params["handedness"] = handedness
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            return data.get("data", [])
        
        log.error(f"API error: {data.get('message')}")
        return []
    
    except Exception as e:
        log.error(f"Failed to fetch batted balls: {e}")
        return []
