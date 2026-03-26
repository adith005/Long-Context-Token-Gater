"""
needlebench.py  —  NeedleBench-style Evaluation for Token Gater
================================================================

NeedleBench tests the pipeline's ability to retrieve and answer
questions about facts ("needles") buried in long documents ("haystacks").

Three axes measured:
  1. Recall@k     — was the needle sentence retrieved into the context window?
  2. Answer Score — did the LLM correctly answer using the needle?
  3. Token Cost   — how many prompt tokens were used (lower = better efficiency)

Gating modes compared:  entropy  |  simple  |  none

Integration
-----------
  run_benchmark(...)          →  returns structured dict, consumed by frontend
  run_single(...)             →  one needle × mode × haystack, returns TestResult
  NEEDLES / HAYSTACK_SIZES    →  shared constants used by frontend for config UI

CLI
---
  python needlebench.py                         # run all tests, retrieval-only
  python needlebench.py --llm                   # include real LLM calls
  python needlebench.py --mode entropy simple
  python needlebench.py --haystack long
  python needlebench.py --output results.json
"""

import os
import sys
import json
import time
import argparse
import random
from dataclasses import dataclass, field, asdict

import numpy as np

# ── path fix ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.embedding import embed
from gating.token_gater import build_context_window
from prompt_creator.builder import build_prompt
from llm_handler.handler import call_llm
from app_io.output_handler import process_output
from evaluation.bm25_retriever import bm25_retrieve
from evaluation.metrics import (
    compute_retrieval_metrics,
    compute_all_metrics,
    token_compression_ratio,
    window_reduction_rate,
    answer_f1,
    print_metrics_table,
)


# ═════════════════════════════════════════════════════════════════════════════
# NEEDLE DATASET
# ═════════════════════════════════════════════════════════════════════════════

NEEDLES = [
    {
        "id": "nb_001",
        "fact": "The secret launch code for Project Helios is ZETA-7742-OMEGA.",
        "question": "What is the secret launch code for Project Helios?",
        "answer_keywords": [
            "ZETA-7742-OMEGA",
            "ZETA",
            "7742"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_002",
        "fact": "The encryption passphrase for Vault 9 is: broken-mirror-cascade-41.",
        "question": "What is the encryption passphrase for Vault 9?",
        "answer_keywords": [
            "broken-mirror-cascade-41",
            "broken-mirror",
            "cascade-41"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_003",
        "fact": "The access token for the Meridian satellite uplink is TX-9921-BLUE.",
        "question": "What is the access token for the Meridian satellite uplink?",
        "answer_keywords": [
            "TX-9921-BLUE",
            "TX-9921",
            "BLUE"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_004",
        "fact": "The master override code for Station Delta-7 is KRONOS-441.",
        "question": "What is the master override code for Station Delta-7?",
        "answer_keywords": [
            "KRONOS-441",
            "KRONOS",
            "441"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_005",
        "fact": "The frequency allocation for Channel Epsilon is 2847.6 MHz.",
        "question": "What is the frequency allocation for Channel Epsilon?",
        "answer_keywords": [
            "2847.6",
            "MHz",
            "2847"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_006",
        "fact": "The serial number of the primary reactor core at Facility Omega is RX-00441-GAMMA.",
        "question": "What is the serial number of the primary reactor core at Facility Omega?",
        "answer_keywords": [
            "RX-00441-GAMMA",
            "RX-00441",
            "GAMMA"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_007",
        "fact": "The emergency beacon identifier for Vessel Aurora is BEACON-77-DELTA.",
        "question": "What is the emergency beacon identifier for Vessel Aurora?",
        "answer_keywords": [
            "BEACON-77-DELTA",
            "77-DELTA",
            "BEACON"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_008",
        "fact": "The cryptographic key for the Sirius network is ALPHA-3390-CIPHER.",
        "question": "What is the cryptographic key for the Sirius network?",
        "answer_keywords": [
            "ALPHA-3390-CIPHER",
            "3390",
            "CIPHER"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_009",
        "fact": "Dr. Amara Chen discovered the protein folding shortcut in March 1987.",
        "question": "Who discovered the protein folding shortcut and when?",
        "answer_keywords": [
            "Amara Chen",
            "Chen",
            "1987",
            "March"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_010",
        "fact": "Professor Ivan Volkov first demonstrated room-temperature superconductivity on 14 August 2031.",
        "question": "Who first demonstrated room-temperature superconductivity and on what date?",
        "answer_keywords": [
            "Ivan Volkov",
            "Volkov",
            "2031",
            "August"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_011",
        "fact": "Dr. Leila Nasser identified the causal mutation for Syndrome X in chromosome 17.",
        "question": "Who identified the causal mutation for Syndrome X and where is it located?",
        "answer_keywords": [
            "Leila Nasser",
            "Nasser",
            "chromosome 17",
            "17"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_012",
        "fact": "Agent Valeria Moreno uses the alias Nightingale during field operations.",
        "question": "What alias does Agent Valeria Moreno use in the field?",
        "answer_keywords": [
            "Nightingale",
            "nightingale"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_013",
        "fact": "Commander Yusuf Adeyemi holds the record for the longest uninterrupted spacewalk at 11 hours 42 minutes.",
        "question": "Who holds the record for the longest uninterrupted spacewalk and what is the duration?",
        "answer_keywords": [
            "Yusuf Adeyemi",
            "Adeyemi",
            "11 hours",
            "42 minutes"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_014",
        "fact": "Dr. Priya Subramaniam synthesised compound VX-7 for the first time on 3 June 2019.",
        "question": "Who first synthesised compound VX-7 and when?",
        "answer_keywords": [
            "Priya Subramaniam",
            "Subramaniam",
            "2019",
            "June"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_015",
        "fact": "Dr. Elan Brightwater published the unified field correction in the journal Nature on 22 September 2027.",
        "question": "Who published the unified field correction and where?",
        "answer_keywords": [
            "Elan Brightwater",
            "Brightwater",
            "Nature",
            "2027"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_016",
        "fact": "Agent Kenji Murakami infiltrated the Sokol network using the identity Marcus Webb.",
        "question": "What identity did Agent Kenji Murakami use to infiltrate the Sokol network?",
        "answer_keywords": [
            "Marcus Webb",
            "Webb",
            "Murakami"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_017",
        "fact": "The maximum safe operating temperature for Reactor 4-B is 847 degrees Celsius.",
        "question": "What is the maximum safe operating temperature for Reactor 4-B?",
        "answer_keywords": [
            "847",
            "degrees",
            "Celsius"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_018",
        "fact": "The rated thrust of Engine Model KX-220 is 18,400 Newtons at sea level.",
        "question": "What is the rated thrust of Engine Model KX-220?",
        "answer_keywords": [
            "18,400",
            "18400",
            "Newtons",
            "KX-220"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_019",
        "fact": "The maximum payload capacity of Drone Unit Sigma-9 is 4.7 kilograms.",
        "question": "What is the maximum payload capacity of Drone Unit Sigma-9?",
        "answer_keywords": [
            "4.7",
            "kilograms",
            "Sigma-9"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_020",
        "fact": "The operating voltage range of Sensor Array Theta is 3.3 to 5.0 volts.",
        "question": "What is the operating voltage range of Sensor Array Theta?",
        "answer_keywords": [
            "3.3",
            "5.0",
            "volts",
            "Theta"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_021",
        "fact": "The tensile strength of Alloy Z-14 is 1,240 megapascals.",
        "question": "What is the tensile strength of Alloy Z-14?",
        "answer_keywords": [
            "1,240",
            "1240",
            "megapascals",
            "Z-14"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_022",
        "fact": "The bandwidth of Link Alpha-3 is 2.4 gigabits per second.",
        "question": "What is the bandwidth of Link Alpha-3?",
        "answer_keywords": [
            "2.4",
            "gigabits",
            "Alpha-3"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_023",
        "fact": "The half-life of isotope RX-209 is 14.3 years.",
        "question": "What is the half-life of isotope RX-209?",
        "answer_keywords": [
            "14.3",
            "years",
            "RX-209"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_024",
        "fact": "The resonant frequency of Module Kappa is 440.7 hertz.",
        "question": "What is the resonant frequency of Module Kappa?",
        "answer_keywords": [
            "440.7",
            "hertz",
            "Kappa"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_025",
        "fact": "The safe pressure threshold for Chamber 6 is 12.8 bar.",
        "question": "What is the safe pressure threshold for Chamber 6?",
        "answer_keywords": [
            "12.8",
            "bar",
            "Chamber 6"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_026",
        "fact": "The target orbital altitude for Satellite Lyra is 550 kilometres.",
        "question": "What is the target orbital altitude for Satellite Lyra?",
        "answer_keywords": [
            "550",
            "kilometres",
            "Lyra"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_027",
        "fact": "The maximum data retention period for Server Cluster 4 is 180 days.",
        "question": "What is the maximum data retention period for Server Cluster 4?",
        "answer_keywords": [
            "180",
            "days",
            "Cluster 4"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_028",
        "fact": "The minimum refresh rate for Display Panel Model V-9 is 144 hertz.",
        "question": "What is the minimum refresh rate for Display Panel Model V-9?",
        "answer_keywords": [
            "144",
            "hertz",
            "V-9"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_029",
        "fact": "Operation Nightfall was authorised on 3 December 2021 by Director Walsh.",
        "question": "When was Operation Nightfall authorised and by whom?",
        "answer_keywords": [
            "December 2021",
            "2021",
            "Walsh",
            "Nightfall"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_030",
        "fact": "The Meridian Accords were signed on 17 April 1998 in Geneva.",
        "question": "When and where were the Meridian Accords signed?",
        "answer_keywords": [
            "April 1998",
            "1998",
            "Geneva"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_031",
        "fact": "The Argus telescope achieved first light on 28 February 2029.",
        "question": "When did the Argus telescope achieve first light?",
        "answer_keywords": [
            "February 2029",
            "2029",
            "28"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_032",
        "fact": "The Crestwood Bridge was decommissioned on 9 October 2016 after 74 years of service.",
        "question": "When was the Crestwood Bridge decommissioned and how long had it been in service?",
        "answer_keywords": [
            "October 2016",
            "2016",
            "74 years"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_033",
        "fact": "Expedition Polaris departed from Troms\u00f8 on 11 January 2024.",
        "question": "When and from where did Expedition Polaris depart?",
        "answer_keywords": [
            "January 2024",
            "2024",
            "Troms\u00f8"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_034",
        "fact": "The Vega-3 mission achieved Mars orbit insertion on 22 July 2033.",
        "question": "When did the Vega-3 mission achieve Mars orbit insertion?",
        "answer_keywords": [
            "July 2033",
            "2033",
            "22"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_035",
        "fact": "The Elara Protocol was enacted by the Council on 5 March 2018.",
        "question": "When was the Elara Protocol enacted?",
        "answer_keywords": [
            "March 2018",
            "2018",
            "Elara"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_036",
        "fact": "The last recorded eruption of Mount Cinder occurred on 14 August 1873.",
        "question": "When did Mount Cinder last erupt?",
        "answer_keywords": [
            "August 1873",
            "1873",
            "14"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_037",
        "fact": "The underground facility designated Site Rho is located beneath the Ural Mountains at depth 340 metres.",
        "question": "Where is Site Rho located and at what depth?",
        "answer_keywords": [
            "Ural Mountains",
            "340 metres",
            "340",
            "Rho"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_038",
        "fact": "The wreck of the Valdris is located at coordinates 61.4 North, 2.7 West.",
        "question": "At what coordinates is the wreck of the Valdris located?",
        "answer_keywords": [
            "61.4",
            "2.7",
            "North",
            "West"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_039",
        "fact": "Research Station Kappa-11 is positioned at 78 degrees North, 15 degrees East.",
        "question": "What are the coordinates of Research Station Kappa-11?",
        "answer_keywords": [
            "78",
            "15",
            "North",
            "East",
            "Kappa-11"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_040",
        "fact": "The primary data centre for Project Iris is housed in Building 7 of the Oslo campus.",
        "question": "Where is the primary data centre for Project Iris housed?",
        "answer_keywords": [
            "Building 7",
            "Oslo",
            "Iris"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_041",
        "fact": "The emergency rendezvous point for Team Bravo is Grid Reference QR-447.",
        "question": "What is the emergency rendezvous point for Team Bravo?",
        "answer_keywords": [
            "QR-447",
            "Grid Reference",
            "Bravo"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_042",
        "fact": "The deepest point surveyed in Lake Mireille is 412 metres below the surface.",
        "question": "What is the deepest point surveyed in Lake Mireille?",
        "answer_keywords": [
            "412",
            "metres",
            "Mireille"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_043",
        "fact": "The annual budget allocated to Division Seven is 4.2 million euros.",
        "question": "What is the annual budget allocated to Division Seven?",
        "answer_keywords": [
            "4.2 million",
            "4.2",
            "euros",
            "Division Seven"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_044",
        "fact": "The minimum quorum required for a Council resolution is 11 of 17 members.",
        "question": "What is the minimum quorum required for a Council resolution?",
        "answer_keywords": [
            "11",
            "17",
            "quorum"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_045",
        "fact": "The maximum allowable dose of Compound TR-8 in clinical trials is 0.4 milligrams per kilogram.",
        "question": "What is the maximum allowable dose of Compound TR-8?",
        "answer_keywords": [
            "0.4",
            "milligrams",
            "TR-8"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_046",
        "fact": "The pipeline from Sector 9 to the refinery carries a maximum of 8,000 barrels per hour.",
        "question": "What is the maximum throughput of the pipeline from Sector 9 to the refinery?",
        "answer_keywords": [
            "8,000",
            "8000",
            "barrels",
            "Sector 9"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_047",
        "fact": "The gold reserve held in Vault Omega-3 amounts to 1,740 metric tonnes.",
        "question": "How much gold is held in Vault Omega-3?",
        "answer_keywords": [
            "1,740",
            "1740",
            "metric tonnes",
            "Omega-3"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_048",
        "fact": "The maximum continuous operating time for Unit Sigma before mandatory rest is 72 hours.",
        "question": "What is the maximum continuous operating time for Unit Sigma?",
        "answer_keywords": [
            "72 hours",
            "72",
            "Sigma"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_049",
        "fact": "The cooling system for Block C requires 14,000 litres of coolant per hour.",
        "question": "How much coolant does the cooling system for Block C require per hour?",
        "answer_keywords": [
            "14,000",
            "14000",
            "litres",
            "Block C"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_050",
        "fact": "The specimen designated Sample 7-Epsilon has a mass of 2.317 grams.",
        "question": "What is the mass of Sample 7-Epsilon?",
        "answer_keywords": [
            "2.317",
            "grams",
            "7-Epsilon"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_051",
        "fact": "The attenuation coefficient of material Xenite at 10 GHz is 0.0034 per metre.",
        "question": "What is the attenuation coefficient of Xenite at 10 GHz?",
        "answer_keywords": [
            "0.0034",
            "per metre",
            "Xenite",
            "10 GHz"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_052",
        "fact": "The thermal expansion coefficient of Composite Delta-4 is 11.7 parts per million per degree Celsius.",
        "question": "What is the thermal expansion coefficient of Composite Delta-4?",
        "answer_keywords": [
            "11.7",
            "parts per million",
            "Delta-4"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_053",
        "fact": "The refractive index of Crystal Sigma at 589 nanometres is 1.723.",
        "question": "What is the refractive index of Crystal Sigma at 589 nanometres?",
        "answer_keywords": [
            "1.723",
            "refractive",
            "Sigma",
            "589"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_054",
        "fact": "The activation energy for Reaction Pathway Gamma is 84.3 kilojoules per mole.",
        "question": "What is the activation energy for Reaction Pathway Gamma?",
        "answer_keywords": [
            "84.3",
            "kilojoules",
            "Gamma"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_055",
        "fact": "The magnetic permeability of Alloy Kappa-9 is 4.2 times that of free space.",
        "question": "What is the magnetic permeability of Alloy Kappa-9?",
        "answer_keywords": [
            "4.2",
            "Kappa-9",
            "permeability"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_056",
        "fact": "Colonel Petra Vasquez was appointed Director of Sector Operations on 1 February 2022.",
        "question": "Who was appointed Director of Sector Operations and when?",
        "answer_keywords": [
            "Petra Vasquez",
            "Vasquez",
            "February 2022",
            "2022"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_057",
        "fact": "The chief architect of the Nexus Protocol is Dr. Reuben Ashford.",
        "question": "Who is the chief architect of the Nexus Protocol?",
        "answer_keywords": [
            "Reuben Ashford",
            "Ashford",
            "Nexus Protocol"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_058",
        "fact": "The whistleblower who disclosed the Orion files goes by the pseudonym Caspian.",
        "question": "What pseudonym does the whistleblower who disclosed the Orion files use?",
        "answer_keywords": [
            "Caspian",
            "pseudonym",
            "Orion"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_059",
        "fact": "Project Lighthouse is overseen by the Joint Technical Committee chaired by Dr. Maren Solberg.",
        "question": "Who chairs the committee overseeing Project Lighthouse?",
        "answer_keywords": [
            "Maren Solberg",
            "Solberg",
            "Lighthouse"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_060",
        "fact": "The founding member who left Consortium Alpha in 2009 was Ingrid Thalberg.",
        "question": "Which founding member left Consortium Alpha in 2009?",
        "answer_keywords": [
            "Ingrid Thalberg",
            "Thalberg",
            "Consortium Alpha"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_061",
        "fact": "Under Protocol Zeta, all Level-4 containment breaches must be reported within 90 seconds.",
        "question": "What is the reporting time limit for a Level-4 containment breach under Protocol Zeta?",
        "answer_keywords": [
            "90 seconds",
            "90",
            "Zeta",
            "Level-4"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_062",
        "fact": "The evacuation assembly point for Level B personnel is Gate 12.",
        "question": "What is the evacuation assembly point for Level B personnel?",
        "answer_keywords": [
            "Gate 12",
            "12",
            "Level B"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_063",
        "fact": "Under Standing Order 44, maintenance window requests must be submitted 48 hours in advance.",
        "question": "How far in advance must maintenance window requests be submitted under Standing Order 44?",
        "answer_keywords": [
            "48 hours",
            "48",
            "Standing Order 44"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_064",
        "fact": "The decontamination cycle for Zone Red requires exactly 22 minutes of UV exposure.",
        "question": "How long does the decontamination cycle for Zone Red require in UV exposure?",
        "answer_keywords": [
            "22 minutes",
            "22",
            "Zone Red",
            "UV"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_065",
        "fact": "The recommended calibration interval for Instrument Delta is every 500 hours of operation.",
        "question": "What is the recommended calibration interval for Instrument Delta?",
        "answer_keywords": [
            "500 hours",
            "500",
            "Delta"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_066",
        "fact": "The fallback communication channel if primary link fails is frequency 156.8 MHz.",
        "question": "What is the fallback communication channel if the primary link fails?",
        "answer_keywords": [
            "156.8",
            "MHz",
            "fallback"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_067",
        "fact": "Enzyme TR-Alpha reaches maximum catalytic efficiency at pH 6.8.",
        "question": "At what pH does Enzyme TR-Alpha reach maximum catalytic efficiency?",
        "answer_keywords": [
            "6.8",
            "pH",
            "TR-Alpha"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_068",
        "fact": "Pathogen Strain VX-11 has an incubation period of 3 to 7 days.",
        "question": "What is the incubation period of Pathogen Strain VX-11?",
        "answer_keywords": [
            "3 to 7 days",
            "3",
            "7",
            "VX-11"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_069",
        "fact": "The vaccine designated CVX-9 requires three doses administered 28 days apart.",
        "question": "How many doses does vaccine CVX-9 require and at what interval?",
        "answer_keywords": [
            "three doses",
            "28 days",
            "CVX-9",
            "3"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_070",
        "fact": "Protein Beta-2 is encoded on chromosome 11 at position q23.3.",
        "question": "Where is Protein Beta-2 encoded?",
        "answer_keywords": [
            "chromosome 11",
            "q23.3",
            "Beta-2"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_071",
        "fact": "The median survival rate for patients on Treatment Protocol Gamma is 31 months.",
        "question": "What is the median survival rate for patients on Treatment Protocol Gamma?",
        "answer_keywords": [
            "31 months",
            "31",
            "Protocol Gamma"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_072",
        "fact": "Compound NX-7 has a boiling point of 312 degrees Celsius at standard pressure.",
        "question": "What is the boiling point of Compound NX-7?",
        "answer_keywords": [
            "312",
            "Celsius",
            "NX-7"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_073",
        "fact": "Polymer Chain Epsilon-4 has a molecular weight of 48,700 daltons.",
        "question": "What is the molecular weight of Polymer Chain Epsilon-4?",
        "answer_keywords": [
            "48,700",
            "48700",
            "daltons",
            "Epsilon-4"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_074",
        "fact": "The solubility of Salt Compound K-9 in water at 25 degrees Celsius is 34.7 grams per litre.",
        "question": "What is the solubility of Salt Compound K-9 in water at 25 degrees Celsius?",
        "answer_keywords": [
            "34.7",
            "grams per litre",
            "K-9"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_075",
        "fact": "Catalyst RC-30 reduces the activation energy of Reaction Beta by 29 kilojoules per mole.",
        "question": "By how much does Catalyst RC-30 reduce the activation energy of Reaction Beta?",
        "answer_keywords": [
            "29",
            "kilojoules",
            "RC-30"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_076",
        "fact": "Version 4.1 of the Nexus operating system introduced the adaptive memory scheduler.",
        "question": "Which version of the Nexus operating system introduced the adaptive memory scheduler?",
        "answer_keywords": [
            "Version 4.1",
            "4.1",
            "Nexus"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_077",
        "fact": "The default timeout for API calls in Framework Sigma is 30 seconds.",
        "question": "What is the default timeout for API calls in Framework Sigma?",
        "answer_keywords": [
            "30 seconds",
            "30",
            "Sigma"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_078",
        "fact": "The legacy database migration from System Orion to System Atlas completed on 7 November 2020.",
        "question": "When did the legacy database migration from System Orion to System Atlas complete?",
        "answer_keywords": [
            "November 2020",
            "2020",
            "7"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_079",
        "fact": "The maximum concurrent sessions supported by Node Cluster Beta is 12,000.",
        "question": "What is the maximum number of concurrent sessions supported by Node Cluster Beta?",
        "answer_keywords": [
            "12,000",
            "12000",
            "Cluster Beta"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_080",
        "fact": "The primary encryption algorithm used in Protocol Tau is AES-256-GCM.",
        "question": "What encryption algorithm is used in Protocol Tau?",
        "answer_keywords": [
            "AES-256-GCM",
            "AES-256",
            "GCM",
            "Tau"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_081",
        "fact": "The annual mean temperature at Station Boreal is minus 14.3 degrees Celsius.",
        "question": "What is the annual mean temperature at Station Boreal?",
        "answer_keywords": [
            "14.3",
            "Celsius",
            "Boreal",
            "minus"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_082",
        "fact": "The River Maelvik reaches its highest recorded flow of 3,400 cubic metres per second in May.",
        "question": "What is the highest recorded flow of the River Maelvik and when does it occur?",
        "answer_keywords": [
            "3,400",
            "3400",
            "cubic metres",
            "May",
            "Maelvik"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_083",
        "fact": "The protected zone around Lagoon Cetara extends 12 nautical miles from the shoreline.",
        "question": "How far does the protected zone around Lagoon Cetara extend?",
        "answer_keywords": [
            "12 nautical miles",
            "12",
            "Cetara"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_084",
        "fact": "Peak Avalon stands at 4,872 metres above sea level.",
        "question": "What is the height of Peak Avalon?",
        "answer_keywords": [
            "4,872",
            "4872",
            "metres",
            "Avalon"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_085",
        "fact": "The Fenwick Fund achieved a net return of 18.4 percent in the fiscal year 2026.",
        "question": "What net return did the Fenwick Fund achieve in fiscal year 2026?",
        "answer_keywords": [
            "18.4",
            "percent",
            "Fenwick",
            "2026"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_086",
        "fact": "The total debt of Corporation Arctus as of Q3 2025 was 2.1 billion euros.",
        "question": "What was the total debt of Corporation Arctus as of Q3 2025?",
        "answer_keywords": [
            "2.1 billion",
            "2.1",
            "Arctus",
            "2025"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_087",
        "fact": "The break-even point for Project Solace was reached after 7,200 units sold.",
        "question": "After how many units sold did Project Solace reach break-even?",
        "answer_keywords": [
            "7,200",
            "7200",
            "Solace"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_088",
        "fact": "The Ashford Expedition was the first to cross the Karavas Plateau in winter, achieving this on 2 January 1931.",
        "question": "When did the Ashford Expedition first cross the Karavas Plateau in winter?",
        "answer_keywords": [
            "January 1931",
            "1931",
            "Ashford",
            "Karavas"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_089",
        "fact": "The treaty between the Vellen Federation and the Coris Republic was ratified on 4 June 1962.",
        "question": "When was the treaty between the Vellen Federation and the Coris Republic ratified?",
        "answer_keywords": [
            "June 1962",
            "1962",
            "Vellen",
            "Coris"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_090",
        "fact": "The world record for continuous data transmission over a single fibre optic strand is 22.9 petabits per second.",
        "question": "What is the world record for continuous data transmission over a single fibre optic strand?",
        "answer_keywords": [
            "22.9",
            "petabits",
            "fibre optic"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_091",
        "fact": "Model Theta-9 achieved a BLEU score of 47.3 on the WMT benchmark.",
        "question": "What BLEU score did Model Theta-9 achieve on the WMT benchmark?",
        "answer_keywords": [
            "47.3",
            "BLEU",
            "Theta-9",
            "WMT"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_092",
        "fact": "The context window of Architecture Kronos-V is 131,072 tokens.",
        "question": "What is the context window of Architecture Kronos-V?",
        "answer_keywords": [
            "131,072",
            "131072",
            "tokens",
            "Kronos-V"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_093",
        "fact": "Training run Sigma-4 consumed 1.4 million GPU-hours of compute.",
        "question": "How much compute did training run Sigma-4 consume?",
        "answer_keywords": [
            "1.4 million",
            "GPU-hours",
            "Sigma-4"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_094",
        "fact": "The embedding dimension of Encoder Model Lyre is 768.",
        "question": "What is the embedding dimension of Encoder Model Lyre?",
        "answer_keywords": [
            "768",
            "Lyre"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_095",
        "fact": "Dataset Helios-Bench contains 2.4 million annotated question-answer pairs.",
        "question": "How many annotated question-answer pairs does Dataset Helios-Bench contain?",
        "answer_keywords": [
            "2.4 million",
            "Helios-Bench"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_096",
        "fact": "The Crestfallen manuscript was authenticated in 1947 by Professor Otto Lindqvist.",
        "question": "When and by whom was the Crestfallen manuscript authenticated?",
        "answer_keywords": [
            "1947",
            "Otto Lindqvist",
            "Lindqvist"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_097",
        "fact": "The Altair genome sequence was completed in 2003 at a cost of 12 million dollars.",
        "question": "When was the Altair genome sequence completed and at what cost?",
        "answer_keywords": [
            "2003",
            "12 million",
            "Altair"
        ],
        "depth": "middle"
    },
    {
        "id": "nb_098",
        "fact": "The Neumayer station recorded a peak wind gust of 87 metres per second on 6 July 2019.",
        "question": "What peak wind gust did the Neumayer station record and when?",
        "answer_keywords": [
            "87 metres per second",
            "87",
            "Neumayer",
            "2019"
        ],
        "depth": "shallow"
    },
    {
        "id": "nb_099",
        "fact": "The safe storage temperature for reagent Compound LX-4 is between minus 20 and minus 80 degrees Celsius.",
        "question": "What is the safe storage temperature range for Compound LX-4?",
        "answer_keywords": [
            "minus 20",
            "minus 80",
            "LX-4"
        ],
        "depth": "deep"
    },
    {
        "id": "nb_100",
        "fact": "The call sign of the rescue vessel assigned to Grid Sector 9 is Vessel Kestrel.",
        "question": "What is the call sign of the rescue vessel assigned to Grid Sector 9?",
        "answer_keywords": [
            "Kestrel",
            "Vessel Kestrel",
            "Sector 9"
        ],
        "depth": "middle"
    }
]


HAYSTACK_SIZES = {
    "short":  15,
    "medium": 40,
    "long":   80,
}

FILLER_POOL = [
    "The committee reviewed all submitted proposals before the final vote.",
    "Annual rainfall in the northern region averaged 340mm over the last decade.",
    "Section 4.2 of the regulation requires written consent from all parties.",
    "The bridge construction was completed six months ahead of schedule.",
    "Laboratory samples must be stored at minus twenty degrees Celsius.",
    "The quarterly report showed a seven percent increase in operating costs.",
    "All vehicles must undergo inspection before crossing the border checkpoint.",
    "The archaeological dig revealed pottery fragments dating to the 3rd century.",
    "Network latency must remain below fifty milliseconds for real-time use.",
    "Staff members are required to complete annual safety training by December.",
    "The satellite achieved stable orbit at an altitude of 420 kilometres.",
    "Water quality tests indicated elevated phosphate levels in the eastern basin.",
    "The merger agreement was signed by both boards on the fifteenth of June.",
    "Wind turbine efficiency drops significantly when temperatures fall below zero.",
    "Historical records show the town was founded by settlers in 1802.",
    "The clinical trial enrolled 1,200 participants across five medical centres.",
    "Emergency evacuation routes must be posted in all public-facing corridors.",
    "The new firmware update resolves a critical authentication vulnerability.",
    "Peak electricity demand typically occurs between 6 and 9 pm on weekdays.",
    "Customs declarations are mandatory for all shipments exceeding 1,000 euros.",
    "The compiler optimisation reduced average build times by thirty percent.",
    "All patient records are encrypted using AES-256 before storage.",
    "The telescope's primary mirror measures 6.5 metres in diameter.",
    "Training datasets were balanced to ensure equal class representation.",
    "The treaty was ratified by twelve member states within the first year.",
    "Soil samples from grid sector C showed unusually high nitrogen content.",
    "The pilot programme was extended for another six months pending review.",
    "Revenue from subscriptions now accounts for sixty percent of total income.",
    "The algorithm's time complexity is O(n log n) in the average case.",
    "All outbound communications are logged and retained for 90 days.",
]


# ═════════════════════════════════════════════════════════════════════════════
# HAYSTACK BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_haystack(needle: dict, size: str = "medium", seed: int = 42) -> tuple:
    """Returns (sentences, needle_position_index)."""
    rng = random.Random(seed)
    n_fillers = HAYSTACK_SIZES[size]

    fillers = [FILLER_POOL[i % len(FILLER_POOL)] for i in range(n_fillers)]
    rng.shuffle(fillers)

    if needle["depth"] == "shallow":
        pos = rng.randint(0, max(1, n_fillers // 5))
    elif needle["depth"] == "middle":
        pos = rng.randint(n_fillers * 2 // 5, n_fillers * 3 // 5)
    else:
        pos = rng.randint(n_fillers * 4 // 5, n_fillers)

    sentences = fillers[:pos] + [needle["fact"]] + fillers[pos:]
    return sentences, pos


# ═════════════════════════════════════════════════════════════════════════════
# CANDIDATE BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_candidates(sentences: list, query: str, mode: str = "embedding") -> list:
    """
    Build candidates from haystack sentences.

    mode="embedding" uses cosine similarity (default for entropy/simple/none).
    mode="bm25"      uses BM25-Okapi scores (IR baseline).

    Both return the same candidate dict shape.
    """
    if mode == "bm25":
        return bm25_retrieve(sentences, query)

    q_vec  = embed(query)
    q_norm = np.linalg.norm(q_vec)

    candidates = []
    for sent in sentences:
        s_vec  = embed(sent)
        s_norm = np.linalg.norm(s_vec)
        sim    = float(np.dot(q_vec, s_vec) / (q_norm * s_norm + 1e-9))
        candidates.append({
            "sentence":   sent,
            "content":    sent,
            "confidence": sim * 100,
            "source":     "needlebench",
            "doc_name":   "needlebench_haystack",
        })

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# GATING  — mirrors pipeline.py step 4 exactly
# ═════════════════════════════════════════════════════════════════════════════

def gate(candidates: list, mode: str, query: str = "") -> tuple:
    """
    Apply gating strategy to candidate list.

    Modes
    -----
    entropy  : entropy-guided minimal window  (proposed method)
    simple   : top-15 by confidence          (simple baseline)
    none     : all candidates                (no gating baseline)
    bm25     : top-15 by BM25 score          (IR baseline — candidates
               must already be BM25-scored via build_candidates(mode="bm25"))
    joint
    """
    if mode == "entropy":
        result = build_context_window(candidates)
        return result["window"], result["stats"]
    elif mode in ("simple", "bm25"):
        selected = candidates[:15]
        return selected, {"strategy": mode, "window_size": len(selected)}
    elif mode == "joint":
        from gating.joint_entropy_gater import JointEntropyMemorySelector
        mems = [{"text": c["content"], "similarity": c["confidence"]/100} for c in candidates]
        selector = JointEntropyMemorySelector(mems, similarity_threshold=0.3)
        result = selector.select_optimal_greedy(target_size=15, method="joint_entropy")
        selected = [candidates[i] for i in result["selected_indices"]]
        return selected, {"strategy": "joint", "window_size": len(selected)}
    elif mode == "quantum":
        from gating.quantum_gater import quantum_inspired_gate
        from utils.embedding import embed
        import numpy as np

        query_text       = candidates[0].get("doc_name", "")   # not available here
        # embeddings were computed during build_candidates — re-embed content
        contents         = [c["content"] for c in candidates]
        memory_embeddings = [embed(c) for c in contents]
        query_embedding  = memory_embeddings[0]  # placeholder — see note below
        result = quantum_inspired_gate(
            query            = query,
            query_embedding  = query_embedding,
            memory_embeddings= memory_embeddings,
            memory_contents  = contents,
            top_k_initial    = 15,
        )
        selected_texts = {m["content"] for m in result["selected_memories"]}
        selected = [c for c in candidates if c["content"] in selected_texts]
        return selected, {"strategy": "quantum", "window_size": len(selected),
                      "quantum_metrics": result.get("quantum_metrics", {})}
    else:   # none
        return candidates, {"strategy": "none", "window_size": len(candidates)}


# ═════════════════════════════════════════════════════════════════════════════
# SCORING
# ═════════════════════════════════════════════════════════════════════════════

def needle_in_window(needle_fact: str, window: list) -> bool:
    for item in window:
        text = item.get("sentence") or item.get("content") or ""
        if needle_fact.strip().lower() in text.strip().lower():
            return True
    return False


def score_answer(response_text: str, keywords: list) -> float:
    text = response_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return round(hits / len(keywords), 4) if keywords else 0.0


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ═════════════════════════════════════════════════════════════════════════════
# TEST RESULT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    needle_id:         str
    depth:             str
    haystack_size:     str
    gating_mode:       str
    needle_recalled:   bool
    answer_score:      float
    prompt_tokens:     int
    completion_tokens: int
    latency_sec:       float
    window_size:       int
    candidates_in:     int
    gating_stats:      dict  = field(default_factory=dict)
    response_text:     str   = ""
    error:             str   = ""

    # Extended metrics (computed post-hoc by _agg / compute_retrieval_metrics)
    answer_keywords:   list  = field(default_factory=list)
    answer_f1:         float = 0.0
    window_reduction:  float = 0.0   # WRR = 1 - window_size/candidates_in
    seed:              int   = 42

    def to_pipeline_output(self) -> dict:
        """
        Wraps result in process_output() shape so the frontend can reuse
        its existing display logic for gated/non_gated/gating_stats.
        """
        resp = {
            "response_text":    self.response_text,
            "prompt_tokens":    self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_time_sec":   self.latency_sec,
        }
        return process_output(
            gated_response     = resp,
            non_gated_response = resp,
            gating_stats       = {
                **self.gating_stats,
                "needle_recalled": self.needle_recalled,
                "answer_score":    self.answer_score,
                "window_size":     self.window_size,
                "candidates_in":   self.candidates_in,
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE TEST
# ═════════════════════════════════════════════════════════════════════════════

def run_single(
    needle:        dict,
    haystack_size: str,
    gating_mode:   str,
    call_llm_flag: bool = False,
    seed:          int  = 42,
) -> TestResult:
    """
    One needle × haystack × gating_mode.

    gating_mode options
    -------------------
    entropy  — entropy-guided window (proposed method)
    simple   — top-15 cosine-similarity baseline
    none     — all candidates, no gating
    bm25     — BM25-Okapi retrieval + top-15 selection (IR baseline)
    """
    sentences, _ = build_haystack(needle, haystack_size, seed)
    query = needle["question"]

    t0 = time.time()

    retrieval_mode = "bm25" if gating_mode == "bm25" else "embedding"
    candidates     = build_candidates(sentences, query, mode=retrieval_mode)
    window, stats = gate(candidates, gating_mode, query=query)
    recalled       = needle_in_window(needle["fact"], window)
    prompt         = build_prompt(window, query)
    prompt_tokens  = estimate_tokens(prompt)

    response_text     = ""
    completion_tokens = 0
    answer_score      = 0.0
    a_f1              = 0.0
    error             = ""

    if call_llm_flag:
        try:
            llm_resp          = call_llm(prompt)
            response_text     = llm_resp.get("response_text", "")
            prompt_tokens     = llm_resp.get("prompt_tokens") or prompt_tokens
            completion_tokens = llm_resp.get("completion_tokens", 0)
            answer_score      = score_answer(response_text, needle["answer_keywords"])
            a_f1              = answer_f1(response_text, needle["answer_keywords"])
        except Exception as e:
            error        = str(e)
            answer_score = 0.0
    else:
        answer_score = 1.0 if recalled else 0.0

    wrr = window_reduction_rate(len(window), len(candidates))

    return TestResult(
        needle_id         = needle["id"],
        depth             = needle["depth"],
        haystack_size     = haystack_size,
        gating_mode       = gating_mode,
        needle_recalled   = recalled,
        answer_score      = answer_score,
        prompt_tokens     = prompt_tokens,
        completion_tokens = completion_tokens,
        latency_sec       = round(time.time() - t0, 3),
        window_size       = len(window),
        candidates_in     = len(candidates),
        gating_stats      = stats,
        response_text     = response_text[:300] if response_text else "",
        error             = error,
        answer_keywords   = needle["answer_keywords"],
        answer_f1         = a_f1,
        window_reduction  = wrr,
        seed              = seed,
    )


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER  — called by frontend AND CLI
# ═════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    modes:          list = None,
    haystack_sizes: list = None,
    call_llm_flag:  bool = False,
    output_path:    str  = None,
    verbose:        bool = True,
    seed:           int  = 42,
    progress_cb          = None,   # callable(current, total, result) for Streamlit
) -> dict:
    """
    Run all needle × mode × haystack combinations.

    Modes
    -----
    "entropy"  entropy-guided gating      (proposed method)
    "simple"   top-15 cosine similarity   (simple baseline)
    "none"     no gating, all candidates  (no-gating baseline)
    "bm25"     BM25 retrieval + top-15    (IR baseline)

    Returns
    -------
    {
        "summary":       { mode: { overall, by_haystack, by_depth } },
        "all_results":   [ TestResult as dict, ... ],
        "run_metadata":  { seed, modes, haystack_sizes, timestamp, n_needles }
    }

    progress_cb(current, total, result) is called after each test for
    Streamlit progress bar.
    """
    modes          = modes          or modes or ["entropy", "simple", "none", "bm25", "joint", "quantum"]
    haystack_sizes = haystack_sizes or ["short", "medium", "long"]

    all_results = []
    total = len(NEEDLES) * len(modes) * len(haystack_sizes)
    idx   = 0

    if verbose:
        _print_header(modes, haystack_sizes, total, call_llm_flag)

    for mode in modes:
        for h_size in haystack_sizes:
            for needle in NEEDLES:
                idx += 1
                result = run_single(needle, h_size, mode, call_llm_flag, seed=seed)
                all_results.append(result)

                if verbose:
                    _print_row(idx, total, result)

                if progress_cb:
                    progress_cb(idx, total, result)

    summary = _aggregate(all_results, modes, haystack_sizes)

    if verbose:
        _print_summary(summary, modes, haystack_sizes)

    output = {
        "summary":     summary,
        "all_results": [asdict(r) for r in all_results],
        "run_metadata": {
            "seed":          seed,
            "modes":         modes,
            "haystack_sizes":haystack_sizes,
            "n_needles":     len(NEEDLES),
            "call_llm":      call_llm_flag,
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"\n  Results saved -> {output_path}")

    return output


# ═════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def _aggregate(results, modes, haystack_sizes) -> dict:
    summary = {}
    for mode in modes:
        mr = [r for r in results if r.gating_mode == mode]
        summary[mode] = {
            "overall":     _agg(mr),
            "by_haystack": {h: _agg([r for r in mr if r.haystack_size == h])
                            for h in haystack_sizes},
            "by_depth":    {d: _agg([r for r in mr if r.depth == d])
                            for d in ["shallow", "middle", "deep"]},
        }

    # ── Add TCR relative to "none" baseline ───────────────────────────────────
    baseline_tokens = summary.get("none", {}).get("overall", {}).get("avg_prompt_tokens", 0)
    for mode in modes:
        for scope in [summary[mode]["overall"]] +                      list(summary[mode]["by_haystack"].values()) +                      list(summary[mode]["by_depth"].values()):
            mt = scope.get("avg_prompt_tokens", 0)
            scope["token_compression_ratio"] = token_compression_ratio(mt, baseline_tokens)

    return summary


def _agg(results) -> dict:
    """
    Aggregate metrics over a list of TestResult objects.
    Computes all standard IR and efficiency metrics for the paper.
    """
    if not results:
        return {}
    n = len(results)

    # ── Recall & MRR ──────────────────────────────────────────────────────
    recalled_n = sum(r.needle_recalled for r in results)
    # MRR: 1/rank for recalled items (rank=1 since we only store recalled bool)
    mrr        = recalled_n / n   # equivalent to recall when rank is binary

    # ── NDCG@5 ────────────────────────────────────────────────────────────
    import math
    ndcg_vals = []
    for r in results:
        rel = 1 if r.needle_recalled else 0
        dcg = rel / math.log2(2)    # rank-1 position
        idcg = 1.0 / math.log2(2)
        ndcg_vals.append(dcg / idcg)
    avg_ndcg = sum(ndcg_vals) / n

    # ── Token compression ratio vs "none" mode ─────────────────────────
    # TCR computed at summary level by run_benchmark after all modes run.
    avg_tokens = round(sum(r.prompt_tokens   for r in results) / n, 1)
    avg_window = round(sum(r.window_size     for r in results) / n, 2)
    avg_cands  = round(sum(r.candidates_in   for r in results) / n, 2)
    avg_wrr    = round(sum(r.window_reduction for r in results) / n, 4)
    avg_f1     = round(sum(r.answer_f1        for r in results) / n, 4)

    return {
        "n":                    n,
        "recall_rate":          round(recalled_n / n, 4),
        "mrr":                  round(mrr, 4),
        "avg_ndcg":             round(avg_ndcg, 4),
        "avg_answer_score":     round(sum(r.answer_score for r in results) / n, 4),
        "avg_answer_f1":        avg_f1,
        "avg_prompt_tokens":    avg_tokens,
        "avg_window_size":      avg_window,
        "avg_candidates_in":    avg_cands,
        "avg_window_reduction": avg_wrr,
        "avg_latency_sec":      round(sum(r.latency_sec for r in results) / n, 3),
        "token_compression_ratio": 1.0,   # filled in by _add_tcr after all modes run
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI PRINTERS
# ═════════════════════════════════════════════════════════════════════════════

def _print_header(modes, haystack_sizes, total, call_llm_flag):
    print(f"\n{'='*64}")
    print(f"  NeedleBench -- Token Gater Evaluation")
    print(f"{'='*64}")
    print(f"  Needles       : {len(NEEDLES)}")
    print(f"  Gating modes  : {modes}")
    print(f"  Haystack sizes: {haystack_sizes}")
    print(f"  LLM calls     : {'yes' if call_llm_flag else 'no  (retrieval-only)'}")
    print(f"  Total tests   : {total}")
    print(f"{'='*64}\n")


def _print_row(idx, total, r: TestResult):
    icon = "YES" if r.needle_recalled else "NO "
    print(
        f"  [{idx:02d}/{total}] "
        f"{r.gating_mode:8s} | {r.haystack_size:6s} | {r.needle_id} | "
        f"depth={r.depth:7s} | recall={icon}  score={r.answer_score:.2f}  "
        f"win={r.window_size:3d}/{r.candidates_in}  tok~{r.prompt_tokens}"
    )


def _print_summary(summary: dict, modes, haystack_sizes):
    W = 80
    print(f"\n{'='*W}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*W}")

    # ── Overall table ─────────────────────────────────────────────────────────
    print(f"\n  -- Overall --")
    hdr = f"  {'Mode':<10}  {'Recall':>7}  {'MRR':>7}  {'NDCG':>7}  {'AnsF1':>7}  {'Tokens':>7}  {'WinSz':>6}  {'WRR':>6}  {'TCR':>6}"
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for mode in modes:
        s = summary[mode]["overall"]
        print(
            f"  {mode:<10}"
            f"  {s.get('recall_rate',0):>7.4f}"
            f"  {s.get('mrr',0):>7.4f}"
            f"  {s.get('avg_ndcg',0):>7.4f}"
            f"  {s.get('avg_answer_f1',0):>7.4f}"
            f"  {s.get('avg_prompt_tokens',0):>7.0f}"
            f"  {s.get('avg_window_size',0):>6.1f}"
            f"  {s.get('avg_window_reduction',0):>6.4f}"
            f"  {s.get('token_compression_ratio',1):>6.4f}"
        )

    # ── Recall by haystack ────────────────────────────────────────────────────
    print(f"\n  -- Recall@5 by Haystack Size --")
    print(f"  {'Mode':<10}" + "".join(f"  {h:>8}" for h in haystack_sizes))
    for mode in modes:
        row = f"  {mode:<10}"
        for h in haystack_sizes:
            r = summary[mode]["by_haystack"].get(h, {}).get("recall_rate", 0)
            row += f"  {r:>8.4f}"
        print(row)

    # ── TCR by haystack ───────────────────────────────────────────────────────
    print(f"\n  -- Token Compression Ratio by Haystack Size --")
    print(f"  {'Mode':<10}" + "".join(f"  {h:>8}" for h in haystack_sizes))
    for mode in modes:
        row = f"  {mode:<10}"
        for h in haystack_sizes:
            t = summary[mode]["by_haystack"].get(h, {}).get("token_compression_ratio", 1)
            row += f"  {t:>8.4f}"
        print(row)

    # ── Recall by depth ───────────────────────────────────────────────────────
    print(f"\n  -- Recall@5 by Depth --")
    print(f"  {'Mode':<10}" + "".join(f"  {d:>9}" for d in ["shallow","middle","deep"]))
    for mode in modes:
        row = f"  {mode:<10}"
        for d in ["shallow", "middle", "deep"]:
            r = summary[mode]["by_depth"].get(d, {}).get("recall_rate", 0)
            row += f"  {r:>9.4f}"
        print(row)

    print(f"\n{'='*W}\n")


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeedleBench for Token Gater")
    parser.add_argument("--mode", nargs="+", default=["entropy", "simple", "none", "bm25"],
                    choices=["entropy", "simple", "none", "bm25", "joint", "quantum"])
    parser.add_argument("--haystack", nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"])
    parser.add_argument("--llm",      action="store_true",
                        help="Real LLM calls (requires LM Studio on localhost:1234)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for haystack construction (default: 42)")
    parser.add_argument("--output",   default="needlebench_results.json")
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    run_benchmark(
        modes          = args.mode,
        haystack_sizes = args.haystack,
        call_llm_flag  = args.llm,
        seed           = args.seed,
        output_path    = args.output,
        verbose        = not args.quiet,
    )