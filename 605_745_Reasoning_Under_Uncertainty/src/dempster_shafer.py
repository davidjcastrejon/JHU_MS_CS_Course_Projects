# Author: David Castrejon
# ENG 605.745.81 Reasoning Under Uncertainty
# Dempster-Shafer Medical Diagnosis Example

from itertools import product

# -----------------------------------------------------------------------------
# Utility: Print mass functions
# -----------------------------------------------------------------------------
def print_mass(m):
    # Display a mass function with formatted keys and values
    for k, v in m.items():
        print(f"{k}: {v:.3f}")
    print()


# ----------------------------------------------------------------
# Dempster's Rule of Combination
# ----------------------------------------------------------------
def combine_mass(m1, m2):
    # Implements Dempster's rule of combination for two mass functions
    combined = {}
    conflict = 0.0

    for (k1, v1), (k2, v2) in product(m1.items(), m2.items()):
        intersect = tuple(sorted(set(k1) & set(k2)))
        if intersect:
            combined[intersect] = combined.get(intersect, 0.0) + v1 * v2
        else:
            conflict += v1 * v2

    if conflict < 1:
        for k in combined:
            combined[k] /= (1 - conflict)
    else:
        raise ValueError("Total conflict: cannot combine (K = 1).")

    return combined


# --------------------------------------------------------------
# Marginalization
# --------------------------------------------------------------
def marginalize_mass(m, target_vars):
    # Compute the marginal mass function over a subset of variables
    marginalized = {}
    for subset, mass in m.items():
        proj = tuple(v for v in subset if v in target_vars)
        marginalized[proj] = marginalized.get(proj, 0.0) + mass
    return marginalized


# ---------------------------------------------------------
# Belief and Plausibility
# ---------------------------------------------------------
def belief(m, A):
    """Compute the belief of a set A given mass function m."""
    bel = 0.0
    set_A = set(A)
    for subset, mass in m.items():
        if set(subset).issubset(set_A):
            bel += mass
    return bel


def plausibility(m, A):
    # Compute the plausibility of a set A given mass function m
    pl = 0.0
    set_A = set(A)
    for subset, mass in m.items():
        if set(subset) & set_A:
            pl += mass
    return pl


# -----------------------------------------------------------------------------
# Medical Diagnosis Example
# -----------------------------------------------------------------------------
def medical_diagnosis_example():
    """
    Diseases: Flu, Cold
    Symptoms: Fever, Cough
    """
    print("=== Medical Diagnosis Example ===\n")

    # ------------------------------
    # Evidence on Diseases
    # ------------------------------
    m_flu = {('Flu',): 0.1, ('No Flu',): 0.8, ('Flu', 'No Flu'): 0.1}
    m_cold = {('Cold',): 0.2, ('No Cold',): 0.7, ('Cold', 'No Cold'): 0.1}

    print("Mass for Flu:")
    print_mass(m_flu)
    print("Mass for Cold:")
    print_mass(m_cold)

    # Combine evidence on same variable
    combined_flu = combine_mass(m_flu, m_flu)
    combined_cold = combine_mass(m_cold, m_cold)

    print("Combined mass for Flu:")
    print_mass(combined_flu)
    print("Combined mass for Cold:")
    print_mass(combined_cold)

    # ----------------------------------------------
    # Conditional Evidence: Symptoms given Diseases
    # ----------------------------------------------
    # Fever depends on Flu
    m_fever_given_flu = {
        ('Fever',): 0.7,
        ('No Fever',): 0.2,
        ('Fever', 'No Fever'): 0.1
    }

    # Cough depends on Cold
    m_cough_given_cold = {
        ('Cough',): 0.6,
        ('No Cough',): 0.3,
        ('Cough', 'No Cough'): 0.1
    }

    # --------------------------------------
    # Propagate Disease Belief to Symptoms
    # --------------------------------------
    # Combine disease belief with symptom evidence
    def propagate(m_disease, m_symptom, disease_present):
        """
        Scale symptom mass by belief that disease is present.
        """
        scale = belief(m_disease, (disease_present,))
        propagated = {}
        for subset, mass in m_symptom.items():
            propagated[subset] = mass * scale
        return propagated

    fever_from_flu = propagate(combined_flu, m_fever_given_flu, 'Flu')
    cough_from_cold = propagate(combined_cold, m_cough_given_cold, 'Cold')

    print("Propagated mass for Fever (from Flu):")
    print_mass(fever_from_flu)
    print("Propagated mass for Cough (from Cold):")
    print_mass(cough_from_cold)

    # --------------------------------------
    # Belief & Plausibility for Symptoms
    # --------------------------------------
    print("Belief and Plausibility for Fever and Cough:\n")
    for symptom_mass, symptom_name in [(fever_from_flu, 'Fever'), (cough_from_cold, 'Cough')]:
        b = belief(symptom_mass, (symptom_name,))
        p = plausibility(symptom_mass, (symptom_name,))
        print(f"{symptom_name}: Belief = {b:.3f}, Plausibility = {p:.3f}")

    print("\n=== End of Example ===\n")


# ------------------------------------
# Entry Point
# ------------------------------------
if __name__ == "__main__":
    medical_diagnosis_example()
