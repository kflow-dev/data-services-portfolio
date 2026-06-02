"""MyMedicine — Content-based medicine recommender with rule filtering.

Uses symptom matching and medical rules to recommend appropriate medications
for travel and general health needs.

Usage:
    CLI:      python cli.py search "fever, diarrhea, headache" --destination Spain
    Streamlit: streamlit run streamlit_app.py
"""

import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import typer

app = typer.Typer(help="MyMedicine: Content-based medicine recommendations.")

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Medicine:
    """Represents a medicine."""
    id: str
    generic_name: str
    brand_names: List[str]
    category: str  # pain_relief, anti_diarrheal, anti_emetic, antihistamine, antibiotic
    indications: List[str]
    dosage: str
    warnings: List[str]
    available_by_prescription: bool


@dataclass
class TravelDestination:
    """Represents a travel destination."""
    country: str
    required_vaccinations: List[str]
    recommended_vaccinations: List[str]
    common_health_risks: List[str]
    medicine_availability: str  # easy, moderate, difficult, restricted


# ============================================================================
# MEDICINE CATALOG
# ============================================================================

def get_medicine_catalog() -> List[Medicine]:
    """Get medicine catalog."""
    medicines = [
        Medicine(
            id="m001",
            generic_name="Paracetamol/Acetaminophen",
            brand_names=["Tylenol", "Panadol", "Calpol"],
            category="pain_relief",
            indications=["fever", "headache", "muscle_pain", "back_pain"],
            dosage="500mg every 4-6 hours, max 4000mg/day",
            warnings=["Avoid with alcohol", "Liver damage risk if overdose"],
            available_by_prescription=False,
        ),
        Medicine(
            id="m002",
            generic_name="Ibuprofen",
            brand_names=["Advil", "Motrin", "Nurofen"],
            category="pain_relief",
            indications=["fever", "headache", "muscle_pain", "inflammation"],
            dosage="200-400mg every 6-8 hours, max 1200mg/day OTC",
            warnings=["Take with food", "Stomach irritation risk"],
            available_by_prescription=False,
        ),
        Medicine(
            id="m003",
            generic_name="Loperamide",
            brand_names=["Imodium", "Diarrest"],
            category="anti_diarrheal",
            indications=["diarrhea", "travelers_diarrhea"],
            dosage="2mg after each loose stool, max 8mg/day",
            warnings=["Do not use if fever or blood in stool"],
            available_by_prescription=False,
        ),
        Medicine(
            id="m004",
            generic_name="Oral Rehydration Salts",
            brand_names=["ORS", "Rehidrat"],
            category="rehydration",
            indications=["dehydration", "diarrhea", "vomiting"],
            dosage="Mix with water as directed, drink as needed",
            warnings=["Use clean water"],
            available_by_prescription=False,
        ),
        Medicine(
            id="m005",
            generic_name="Dimenhydrinate",
            brand_names=["Dramamine", "Gravol"],
            category="anti_emetic",
            indications=["motion_sickness", "nausea", "vertigo"],
            dosage="50mg every 4-6 hours, max 400mg/day",
            warnings=["May cause drowsiness"],
            available_by_prescription=False,
        ),
        Medicine(
            id="m006",
            generic_name="Cetirizine",
            brand_names=["Zyrtec", "Alergia"],
            category="antihistamine",
            indications=["allergies", "hay_fever", "hives", "itching"],
            dosage="10mg once daily",
            warnings=["May cause mild drowsiness"],
            available_by_prescription=False,
        ),
        Medicine(
            id="m007",
            generic_name="Amoxicillin",
            brand_names=["Amoxil", "Trimox"],
            category="antibiotic",
            indications=["bacterial_infections", "respiratory_infections", "ear_infections"],
            dosage="500mg every 8 hours for 7-10 days",
            warnings=["Complete full course", "Allergy risk"],
            available_by_prescription=True,
        ),
        Medicine(
            id="m008",
            generic_name="Hydrocortisone Cream",
            brand_names=["Cortaid", "Cortenema"],
            category="topical",
            indications=["skin_rashes", "insect_bites", "eczema", "dermatitis"],
            dosage="Apply thin layer 2-3 times daily",
            warnings=["Do not use on infected skin"],
            available_by_prescription=False,
        ),
        Medicine(
            id="m009",
            generic_name="Antiseptic Solution",
            brand_names=["Betadine", "Dakin"],
            category="topical",
            indications=["wound_care", "disinfection", "pre_surgery"],
            dosage="Apply to affected area",
            warnings=["For external use only"],
            available_by_prescription=False,
        ),
        Medicine(
            id="m010",
            generic_name="Sunscreen SPF 50+",
            brand_names=["Solbar", "Heliocare"],
            category="protection",
            indications=["sun_protection", "sunburn_prevention"],
            dosage="Apply 15min before exposure, reapply every 2 hours",
            warnings=["Use liberally", "Protect children"],
            available_by_prescription=False,
        ),
    ]
    return medicines


def get_destination_health_info() -> Dict[str, TravelDestination]:
    """Get health information for travel destinations."""
    destinations = {
        "spain": TravelDestination(
            country="Spain",
            required_vaccinations=[],
            recommended_vaccinations=["Hepatitis A"],
            common_health_risks=["food_borne_illness", "sun_exposure"],
            medicine_availability="easy",
        ),
        "mexico": TravelDestination(
            country="Mexico",
            required_vaccinations=[],
            recommended_vaccinations=["Hepatitis A", "Typhoid"],
            common_health_risks=["travelers_diarrhea", "sun_exposure", "mosquito_borne"],
            medicine_availability="moderate",
        ),
        "thailand": TravelDestination(
            country="Thailand",
            required_vaccinations=[],
            recommended_vaccinations=["Hepatitis A", "Typhoid", "Rabies (if rural)"],
            common_health_risks=["travelers_diarrhea", "dengue", "malaria (some areas)"],
            medicine_availability="moderate",
        ),
        "brazil": TravelDestination(
            country="Brazil",
            required_vaccinations=["Yellow Fever (Amazon region)"],
            recommended_vaccinations=["Hepatitis A", "Typhoid"],
            common_health_risks=["dengue", "zika", "chikungunya", "travelers_diarrhea"],
            medicine_availability="moderate",
        ),
        "india": TravelDestination(
            country="India",
            required_vaccinations=["Hepatitis A", "Typhoid"],
            recommended_vaccinations=["Hepatitis B", "Rabies", "Japanese Encephalitis (rural)"],
            common_health_risks=["travelers_diarrhea", "dengue", "malaria"],
            medicine_availability="difficult",
        ),
        "japan": TravelDestination(
            country="Japan",
            required_vaccinations=[],
            recommended_vaccinations=["Hepatitis A"],
            common_health_risks=["food_borne_illness"],
            medicine_availability="easy",
        ),
        "egypt": TravelDestination(
            country="Egypt",
            required_vaccinations=[],
            recommended_vaccinations=["Hepatitis A", "Typhoid"],
            common_health_risks=["travelers_diarrhea", "sun_exposure"],
            medicine_availability="moderate",
        ),
        "morocco": TravelDestination(
            country="Morocco",
            required_vaccinations=[],
            recommended_vaccinations=["Hepatitis A", "Typhoid"],
            common_health_risks=["travelers_diarrhea", "sun_exposure"],
            medicine_availability="moderate",
        ),
    }
    return destinations


# ============================================================================
# SYMPTOM PARSING
# ============================================================================

def parse_symptoms(symptoms_text: str) -> List[str]:
    """Parse symptoms text into standardized list."""
    symptoms_map = {
        "fever": ["fever", "hot", "temperature", "high temp"],
        "headache": ["headache", "head pain", "migraine", "throbbing"],
        "diarrhea": ["diarrhea", "stomach upset", "loose stool", "digestive"],
        "nausea": ["nausea", "vomiting", "sick to stomach", "upset stomach"],
        "pain": ["pain", "ache", "sore", "hurt", "aching"],
        "allergies": ["allergies", "allergic", "hay fever", "sneezing", "itchy"],
        "infection": ["infection", "sick", "ill", "infected", "feverish"],
        "skin": ["rash", "skin", "bump", "bite", "itching", "red"],
        "travel": ["travelers", "traveler", "tourist"],
    }

    symptoms_lower = symptoms_text.lower()
    found = []

    for symptom, keywords in symptoms_map.items():
        for keyword in keywords:
            if keyword in symptoms_lower:
                if symptom not in found:
                    found.append(symptom)
                break

    return found


def match_medicines(symptoms: List[str], medicines: List[Medicine]) -> List[Tuple[Medicine, int]]:
    """Match medicines to symptoms based on indications."""
    scores = []

    for medicine in medicines:
        match_count = 0
        matched_indications = []

        for symptom in symptoms:
            for indication in medicine.indications:
                # Direct match
                if symptom == indication:
                    match_count += 2
                    matched_indications.append(indication)
                # Partial match
                elif symptom in indication or indication in symptom:
                    match_count += 1
                    matched_indications.append(indication)

        if match_count > 0:
            scores.append((medicine, match_count, matched_indications))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


# ============================================================================
# RULE-BASED FILTERING
# ============================================================================

def apply_health_rules(medicines: List[Tuple[Medicine, int, List]], destination: TravelDestination) -> List[Tuple[Medicine, int, List]]:
    """Apply health rules based on destination."""
    filtered = []

    for medicine, score, indications in medicines:
        # Adjust scores based on destination availability
        if destination.medicine_availability == "restricted":
            if medicine.available_by_prescription:
                score *= 0.5  # Reduce score for prescription meds in restricted areas

        # Adjust based on common health risks
        for risk in destination.common_health_risks:
            for indication in medicine.indications:
                if risk in indication:
                    score *= 1.2  # Boost for destination-specific needs

        filtered.append((medicine, score, indications))

    return filtered


# ============================================================================
# CLI COMMANDS
# ============================================================================

@app.command()
def search(
    symptoms: str = typer.Argument(..., help="Symptoms or needs (e.g., 'fever, diarrhea, headache')"),
    destination: str = typer.Option(..., "--destination", "-d", help="Travel destination country"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of recommendations"),
    prescription: bool = typer.Option(False, "--prescription", "-p", help="Include prescription medicines"),
):
    """Search for medicines based on symptoms and destination."""
    medicines = get_medicine_catalog()
    destinations = get_destination_health_info()

    destination_info = destinations.get(destination.lower(), destinations["spain"])

    typer.echo(f"Medicine recommendations for: {destination}")
    typer.echo(f"Symptoms: {symptoms}")
    typer.echo()

    # Parse symptoms
    parsed_symptoms = parse_symptoms(symptoms)
    typer.echo(f"Parsed symptoms: {', '.join(parsed_symptoms)}")
    typer.echo()

    # Match medicines
    matches = match_medicines(parsed_symptoms, medicines)

    if not matches:
        typer.echo("No direct matches found.")
        typer.echo("\nGeneral recommendations for travel:")
        for med in medicines[:3]:
            if not med.available_by_prescription or prescription:
                typer.echo(f"  - {med.generic_name}: {', '.join(med.indications[:3])}")
        return

    # Apply destination rules
    filtered_matches = apply_health_rules(matches, destination_info)

    # Show results
    typer.echo(f"Top {top_k} recommendations:\n")
    for i, (medicine, score, indications) in enumerate(filtered_matches[:top_k], 1):
        avail = "Rx required" if medicine.available_by_prescription else "OTC"
        if not prescription and medicine.available_by_prescription:
            continue
        typer.echo(f"{i}. {medicine.generic_name} ({avail})")
        typer.echo(f"   Brands: {', '.join(medicine.brand_names[:3])}")
        typer.echo(f"   Matches: {', '.join(indications[:3])}")
        typer.echo(f"   Dosage: {medicine.dosage}")
        typer.echo(f"   Score: {score}")
        typer.echo()

    # Show destination-specific advice
    typer.echo("Destination health advice:")
    if destination_info.required_vaccinations:
        typer.echo(f"  Required: {', '.join(destination_info.required_vaccinations)}")
    if destination_info.recommended_vaccinations:
        typer.echo(f"  Recommended: {', '.join(destination_info.recommended_vaccinations)}")
    if destination_info.common_health_risks:
        typer.echo(f"  Common risks: {', '.join(destination_info.common_health_risks)}")


@app.command()
def check_availability(
    medicine: str = typer.Argument(..., help="Medicine name"),
    destination: str = typer.Option(..., "--destination", "-d", help="Destination country"),
):
    """Check medicine availability in destination."""
    medicines = get_medicine_catalog()
    destinations = get_destination_health_info()

    medicine_lower = medicine.lower()
    found = None

    for med in medicines:
        if medicine_lower in med.generic_name.lower() or any(medicine_lower in b.lower() for b in med.brand_names):
            found = med
            break

    if not found:
        typer.echo(f"Medicine '{medicine}' not found in catalog.")
        return

    destination_info = destinations.get(destination.lower(), destinations["spain"])

    typer.echo(f"Medicine: {found.generic_name}")
    typer.echo(f"Destination: {destination}")
    typer.echo(f"Availability: {destination_info.medicine_availability}")
    typer.echo(f"Prescription required: {found.available_by_prescription}")

    if destination_info.medicine_availability == "easy":
        typer.echo("Expected: Widely available in pharmacies")
    elif destination_info.medicine_availability == "moderate":
        typer.echo("Expected: Available but may need to search")
    elif destination_info.medicine_availability == "difficult":
        typer.echo("Expected: Limited availability, bring from home")
    else:
        typer.echo("Expected: Restricted import, check customs")


@app.command()
def destination_advice(
    destination: str = typer.Argument(..., help="Travel destination"),
):
    """Get health advice for a travel destination."""
    destinations = get_destination_health_info()

    dest = destinations.get(destination.lower())
    if not dest:
        typer.echo(f"Unknown destination: {destination}")
        typer.echo(f"Available destinations: {', '.join(destinations.keys())}")
        return

    typer.echo(f"Health advice for {dest.country}:")
    typer.echo()
    typer.echo("Vaccinations:")
    if dest.required_vaccinations:
        typer.echo(f"  Required: {', '.join(dest.required_vaccinations)}")
    else:
        typer.echo("  Required: None")
    if dest.recommended_vaccinations:
        typer.echo(f"  Recommended: {', '.join(dest.recommended_vaccinations)}")
    typer.echo()
    typer.echo("Common health risks:")
    for risk in dest.common_health_risks:
        typer.echo(f"  - {risk.replace('_', ' ').title()}")
    typer.echo()
    typer.echo(f"Medicine availability: {dest.medicine_availability}")


if __name__ == "__main__":
    app()
