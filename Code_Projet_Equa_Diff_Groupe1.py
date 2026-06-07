import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

# ==========================================
# FONCTION DE TEST (g) ET FONCTION P
# ==========================================

# Définition des différentes fonctions g

def g_classique(x):
    """Fonction g classique respectant g(0) = g'(0) = 0"""
    return x**3 - x**2


def g_oscillante(x):
    """Fonction g oscillante respectant g(0) = g'(0) = 0"""
    return -((3 * x**2) / 2) * np.cos(x) + ((x**3) / 2) * np.sin(x)

def g_suite(x, n):
    """Fonction g suité respectant g(0) = 0 et g'(0) -> 0"""
    poly  = 3*x**5 - 22.5*x**4 + 48.5*x**3 - 27*x**2
    if np.isinf(n):
        return poly
    gauss = (x**3 - 4*x**2 - 2*x) / (2*n) * np.exp(-(x-4)**2 / 2)
    return poly + gauss

# Calcul de la primitive de g : G

def G(y, g):
    """Calcule la primitive G(y) = intégrale de 0 à y de g(t) dt"""
    valeur, _ = quad(g, 0, y)
    return valeur

def G_suite(y, g, n):
    """Calcule la primitive Gn(y) = intégrale de 0 à y de gn(t) dt"""
    valeur, _ = quad(lambda t: g(t, n), 0, y)
    return valeur

# Calcul de la fonction P

def P(x, func_g):
    """Calcule P(x) selon g"""
    if x == 0:
        return 0.0
    integral, _ = quad(func_g, 0, x)
    return (-2 / (x**2)) * integral

def P_suite(x, func_g, n):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.zeros_like(x)
    for i, xi in enumerate(x):
        if abs(xi) < 1e-10:
            result[i] = 0.0
        else:
            integral, _ = quad(func_g, 0, xi, args=(n,))
            result[i] = (-2 / xi**2) * integral
    return float(result.squeeze())
 
# ==========================================
# PARTIE (i) : ALGORITHME DE RECHERCHE c, µ
# ==========================================

def trouver_c_mu(func_g, x_max=5.0, nb_points=5000):
    """Retourne une liste de tuples (c, µ) valides en vérifiant
    les conditions de l'algorithme."""

    x_vals = np.linspace(1e-4, x_max, nb_points)
    dx = x_vals[1] - x_vals[0]
    p_vals = np.array([P(x, func_g) for x in x_vals])

    peaks, _ = find_peaks(p_vals)
    resultats = []

    for peak_idx in peaks:
        c_approx = x_vals[peak_idx]

        # Affinage du sommet en limitant la recherche aux voisins directs du pas de grille
        res = minimize_scalar(
            lambda x: -P(x, func_g),
            bounds=(max(1e-4, c_approx - 2 * dx), c_approx + 2 * dx),
            method="bounded",
        )
        if res.success:
            c = res.x
            mu = P(c, func_g)

            # Vérification des conditions strictes c > 0, µ > 0 
            if mu > 0 and c > 0:
                x_avant_c = np.linspace(1e-4, c - 1e-5, 200)
                p_avant_c = np.array([P(x, func_g) for x in x_avant_c])

                #Vérification de P(x) < P(c)
                if np.all(p_avant_c < mu):
                    resultats.append((c, mu))

    return resultats

def trouver_c_mu_g_suite(func_g, n, x_max=5.0, nb_points=5000):
    """Retourne une liste de tuples (c, µ) valides en vérifiant
    les conditions de l'algorithme."""

    x_vals = np.linspace(1e-4, x_max, nb_points)
    dx = x_vals[1] - x_vals[0]
    p_vals = np.array([P_suite(x, func_g, n) for x in x_vals])

    peaks, _ = find_peaks(p_vals)
    resultats = []

    for peak_idx in peaks:
        c_approx = x_vals[peak_idx]

        res = minimize_scalar(
            lambda x: -P_suite(x, func_g, n),
            bounds=(max(1e-4, c_approx - 2 * dx), c_approx + 2 * dx),
            method="bounded",
        )
        if res.success:
            c = res.x
            mu = P_suite(c, func_g, n)

            if mu > 0 and c > 0:
                x_avant_c = np.linspace(1e-4, c - 1e-5, 200)
                p_avant_c = np.array([P_suite(x, func_g, n) for x in x_avant_c])

                if np.all(p_avant_c < mu):
                    resultats.append((c, mu))

    return resultats

# ==========================================
# PARTIE (ii) : RÉSOLUTION ET TRACÉ DE Q(x)
# ==========================================

def trouver_solution_Q_droite(mu, c, g, Longueur_x=15, n_points=50000, epsilon=1e-5):
    """
    Résout l'équation avec un événement d'arrêt pour bloquer les oscillations.
    """
    def f(t, Y):
        Q, dQ = Y
        return [dQ, mu * Q + g(Q)]

    # --- AJOUT DE L'ÉVÉNEMENT D'ARRÊT ---
    # On arrête l'intégration dès que Q s'approche de l'asymptote c
    def evenement_atteint_c(t, Y):
        Q, dQ = Y
        return Q - (c - 1e-3) # S'annule quand Q atteint c - 0.001
    
    evenement_atteint_c.terminal = True  # Force l'arrêt du solveur
    evenement_atteint_c.direction = 1   # Uniquement quand Q augmente vers c

    # Conditions initiales en +T
    # On initialise Q en +T
    Q_T = epsilon

    #On initialise dQ en +T
    valeur_G = G(Q_T, g)
    contenu_racine = mu * (Q_T**2) + 2 * valeur_G
    dQ_T = -np.sqrt(max(contenu_racine, 0.0))
    Y0 = [Q_T, dQ_T]
        
    T = Longueur_x
    t_eval = np.linspace(T, -T, n_points)

    # Résolution grâce à solve_ivp avec prise en compte de l'événement
    sol = solve_ivp(
        fun=f,                         # Fonction définissant le système : dy/dt = f(t, y)
        t_span=[T, -T],                # Intervalle d'intégration (intégration rétrograde de T à -T)
        y0=Y0,                         # Conditions initiales à l'instant t = T
        t_eval=t_eval,                 # Instants précis où enregistrer la solution calculée
        events=evenement_atteint_c,    # Fonction(s) pour détecter des conditions d'arrêt ou des événements particuliers
        method="RK45",                 # Méthode d'intégration : Runge-Kutta d'ordre 4(5) à pas adaptatif
        rtol=1e-9,                     # Tolérance relative pour le contrôle de l'erreur
        atol=1e-11                     # Tolérance absolue pour le contrôle de l'erreur
    )
    
    # On inverse les éléments pour que ça aille de -T à T

    t_vals = sol.t[::-1] 
    Q_vals = sol.y[0][::-1]

    # Prolongement par un plateau horizontal parfait (car la solution est stable en c)
    # Si le solveur s'est arrêté plus tôt que -T, on comble le vide à gauche avec la valeur `c`
    if t_vals[0] > -T:
        # On crée 100 points de temps pour combler le "vide" entre la borne -T et le moment où le solveur s'est arrêté
        extension_x = np.linspace(-T, t_vals[0], 100, endpoint=False)
    
        # On crée les valeurs de Q correspondantes, toutes égales à la constante 'c' (puisque le système s'est stabilisé à 'c')
        extension_y = np.full_like(extension_x, c)

        # On colle ces nouveaux points de temps au TOUT DÉBUT de nos anciennes valeurs de temps
        t_vals = np.concatenate([extension_x, t_vals])
    
        # On colle ces nouvelles valeurs de Q au TOUT DÉBUT de nos anciennes valeurs de Q
        Q_vals = np.concatenate([extension_y, Q_vals])

    # Recentrage de la transition à l'origine x = 0

    # On cherche l'indice (la position dans la liste) où la valeur de Q est la plus proche du milieu de la transition (c / 2)
    idx_milieu = np.argmin(np.abs(Q_vals - (c / 2)))

    # On récupère l'instant précis 't' qui correspond à ce milieu de transition
    t_centre = t_vals[idx_milieu]

    # On décale tout l'axe du temps en soustrayant 't_centre' : le moment de la transition devient le nouvel instant t = 0
    t_vals = t_vals - t_centre

    return t_vals, Q_vals


def trouver_solution_Q_g_suite(mu, c, g, n, Longueur_x=15, n_points=50000, epsilon=1e-5):
    """
    Résout l'équation avec un événement d'arrêt pour bloquer les oscillations.
    g est une fonction de la forme g(x, n).
    """
    def f(t, Y):
        Q, dQ = Y
        return [dQ, mu * Q + g(Q, n)]

    def evenement_atteint_c(t, Y):
        Q, dQ = Y
        return Q - (c - 1e-3)
    
    evenement_atteint_c.terminal = True
    evenement_atteint_c.direction = 1

    Q_T = epsilon

    valeur_G = G_suite(Q_T, g, n)
    contenu_racine = mu * (Q_T**2) + 2 * valeur_G
    dQ_T = -np.sqrt(max(contenu_racine, 0.0))
    Y0 = [Q_T, dQ_T]
        
    T = Longueur_x
    t_eval = np.linspace(T, -T, n_points)

    sol = solve_ivp(
        fun=f,
        t_span=[T, -T],
        y0=Y0,
        t_eval=t_eval,
        events=evenement_atteint_c,
        method="RK45",
        rtol=1e-9,
        atol=1e-11
    )
    
    t_vals = sol.t[::-1] 
    Q_vals = sol.y[0][::-1]

    if t_vals[0] > -T:
        extension_x = np.linspace(-T, t_vals[0], 100, endpoint=False)
        extension_y = np.full_like(extension_x, c)
        t_vals = np.concatenate([extension_x, t_vals])
        Q_vals = np.concatenate([extension_y, Q_vals])

    idx_milieu = np.argmin(np.abs(Q_vals - (c / 2)))
    t_centre = t_vals[idx_milieu]
    t_vals = t_vals - t_centre

    return t_vals, Q_vals

def tracer_solution_Q(sol_x, sol_y, c, mu):
    """Dessine une solution Q(x) sur un graphique."""
    plt.figure(figsize=(8, 4))

    # Trace la courbe de la solution Q(x) en bleu avec une épaisseur de ligne de 2.5 et lui donne un nom pour la légende
    plt.plot(sol_x, sol_y, color="#1f77b4", linewidth=2.5, label=r"$Q(x)$")
    # Trace une ligne horizontale pointillée rouge à la hauteur y = c pour représenter l'asymptote vers c
    plt.axhline(c, color="#d62728", linestyle="--", linewidth=2, label=f"Asymptote $c = {c:.2f}$")
    # Trace une ligne horizontale pointillée noire à la hauteur y = 0 pour l'asymptote vers 0 (l'axe des abscisses)
    plt.axhline(0, color="black", linestyle="--", linewidth=2, label=f"Asymptote 0")
    
    plt.title(f"Profil de la solution $Q(x)$ ($\mu = {mu:.3f}$, c = {c:.3f})")
    plt.xlabel("$x$")
    plt.ylabel("$Q(x)$")
    plt.legend()
    plt.grid(alpha = 0.3)
    
    # Aligne et force l'affichage de l'axe des x uniquement entre -10 et 10
    plt.xlim(-10, 10)
    plt.tight_layout()
    plt.show()

def tracer_solution_Q2(sol_x, sol_y, c, mu, label=""):
    """Dessine une solution Q(x) sur le graphique courant sans effacer le reste."""

    #On dessine une solution sur le graphique courant avec le label mis en paramètre
    plt.plot(sol_x, sol_y, linewidth=2, label=label)

    # On ajoute une ligne pointillée discrète pour le plateau c de cette solution
    plt.axhline(c, linestyle="--", alpha=0.5, linewidth=2)


def tracer_fonction_g_P(func_g, func_P, Longueur_x_P=20, Longueur_x_g=20):
    """Dessine une la fonction g et la fonction P sur deux graphiques."""
    # Génère 500 points régulièrement espacés entre -Longueur_x_g et +Longueur_x_g pour l'axe x de g
    x_g_valeurs = np.linspace(-Longueur_x_g, Longueur_x_g, 500)
    # Génère 500 points entre 0.0001 (pour éviter d'avoir un x=0) et Longueur_x_P pour l'axe x de P
    x_P_valeurs = np.linspace(1e-4, Longueur_x_P, 500)
    # Calcule l'image (y) de chaque point x_g_valeurs en appliquant la fonction func_g
    y_g_valeurs = np.array([func_g(x) for x in x_g_valeurs])
    # Calcule l'image (y) pour P en utilisant à la fois les points x_P_valeurs et la fonction func_g
    y_P_valeurs = np.array([func_P(x, func_g) for x in x_P_valeurs])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Sur le premier graphique (à gauche), trace la courbe de g(x) en rouge avec une épaisseur de 2
    ax1.plot(x_g_valeurs, y_g_valeurs, color="#c31111", linewidth=2, label="$g(x)$")
    # Sur le premier graphique, ajoute une ligne horizontale noire et fine à y = 0 pour marquer l'axe des abscisses
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_title("Fonction $g(x)$")
    ax1.grid(True, alpha=0.3)

    # Sur le second graphique (à droite), trace la courbe de P(x) en bleu avec une épaisseur de 2
    ax2.plot(x_P_valeurs, y_P_valeurs, color="#2170a8", linewidth=2, label="$P(x)$")
    # Sur le second graphique, ajoute une ligne horizontale noire et fine à y = 0 pour marquer l'axe des abscisses
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_title("Fonction $P(x)$")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def tracer_P_suite(func_g, P, n, Longueur_x=20):
    """Dessine la fonction P pour un n donné"""
    # Génère 500 points régulièrement espacés entre 0.0001 (pour éviter d'avoir un x=0) et Longueur_x pour l'axe x
    x_P_valeurs = np.linspace(1e-4, Longueur_x, 500)
    # Calcule l'image (y) pour chaque point en appliquant la fonction P avec la fonction func_g et l'indice n
    y_P_valeurs = np.array([P(x, func_g, n) for x in x_P_valeurs])
    
    # Trace la courbe de P(x) avec une épaisseur de 2 et ajoute une étiquette dynamique indiquant la valeur de n pour la légende
    plt.plot(x_P_valeurs, y_P_valeurs, linewidth=2, label=f'$P(x)$ pour $n={n}$')
    # Ajoute une ligne horizontale noire et fine à y = 0 pour matérialiser l'axe des abscisses
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

    plt.title("Fonction $P(x)$ sur $[0, +\infty[$", fontsize=12)
    plt.xlabel("$x$")
    plt.ylabel("$P(x)$")
    plt.grid(True, alpha=0.3)
    plt.legend()

# ==========================================
# EXÉCUTION DU PROJET COMPLET
# ==========================================

if __name__ == "__main__":


    # --- 1. CAS COMPORTEMENT CLASSIQUE ---


    g_utilise = g_classique # fonction g utilisée
    Longueur_x_P = 3 # Longueur x pour le tracé de P
    Longueur_x_g = 10 # Longueur x pour le tracé de g
    Longueur_x_solution = 100 # Longueur x pour le tracé de la solution Q
    x_max_recherche = 100 # Longueur pour rechercher mu et c
    nb_points_recherche = 3000 # nombre de points pour décomposer l'intervalle pour la recherche de mu et c
    nb_points = 20000 # nombre de points pour la solutions
    print("--- Analyse de g_classique ---")
    
    # Appelle la fonction de recherche pour obtenir la liste des couples (c, mu) qui fonctionnent
    parametres_valides = trouver_c_mu(g_utilise, x_max=x_max_recherche, nb_points=nb_points_recherche)

    # Vérifie si la liste renvoyée n'est pas vide (si des paramètres ont été trouvés)
    if parametres_valides:
        # Affiche le nombre total de couples de paramètres valides découverts
        print(f"{len(parametres_valides)} couples de paramètres trouvés.")
        # Parcourt un par un chaque couple (c, mu) trouvé dans la liste
        for c_trouve, mu_trouve in parametres_valides:
            # Affiche les valeurs précises de c et mu trouvées, arrondies à 4 décimales
            print(f"Paramètres trouvés : c = {c_trouve:.4f}, µ = {mu_trouve:.4f}")
            # Trace et affiche les graphiques des fonctions g(x) et P(x) pour ce couple de paramètres
            tracer_fonction_g_P(g_utilise, P, Longueur_x_P=Longueur_x_P, Longueur_x_g=Longueur_x_g)
            # Calcule numériquement les points (x, y) de la solution Q(x) pour ces paramètres
            sol_x, sol_y = trouver_solution_Q_droite(mu_trouve, c_trouve, g_utilise, Longueur_x=Longueur_x_solution, n_points = nb_points)
            # Trace et affiche le graphique du profil de la solution Q(x) avec ses asymptotes
            tracer_solution_Q(sol_x, sol_y, c_trouve, mu_trouve)
    else:
        print("Aucun paramètre valide trouvé pour g_classique.")


    # --- 2. CAS COMPORTEMENT OSCILLANT ---


    g_utilise = g_oscillante # fonction g utilisée
    Longueur_x_P = 60 # Longueur x pour le tracé de P
    Longueur_x_g = 60 # Longueur x pour le tracé de g
    Longueur_x_solution = 100 # Longueur x pour le tracé de la solution Q
    x_max_recherche = 60 # Longueur pour rechercher mu et c
    nb_points_recherche = 3000 # nombre de points pour décomposer l'intervalle pour la recherche de mu et c
    nb_points = 20000 # nombre de points pour la solutions
    print("\n--- Analyse de g_oscillante ---")
    
    # Cherche et récupère la liste des couples (c, mu) valides pour la fonction g actuelle
    parametres_valides = trouver_c_mu(g_utilise, x_max=x_max_recherche, nb_points=nb_points_recherche)

    # Trace les courbes représentatives des fonctions g(x) et P(x) configurées avec les longueurs données
    tracer_fonction_g_P(g_utilise, P, Longueur_x_P=Longueur_x_P, Longueur_x_g=Longueur_x_g)

    # Vérifie si la liste contient au moins un couple de paramètres valides
    if parametres_valides:
        # Affiche dans la console le nombre de combinaisons de paramètres trouvées
        print(f"{len(parametres_valides)} couples de paramètres trouvés.")

        plt.figure(figsize=(10, 5))
        # Ajoute une ligne horizontale pointillée noire d'épaisseur 2 à l'ordonnée y = 0
        plt.axhline(0, color="black", linestyle="--", linewidth=2)

        # Parcourt chaque couple de paramètres trouvé en récupérant son indice et ses valeurs de c et mu
        for idx, (c_trouve, mu_trouve) in enumerate(parametres_valides):
            # Calcule les coordonnées x et y de la solution de l'équation pour le couple actuel
            sol_x, sol_y = trouver_solution_Q_droite(mu_trouve, c_trouve, g_utilise, Longueur_x=Longueur_x_solution, n_points = nb_points)
            # Construit une chaîne de caractères formatée contenant le numéro de la solution, sa valeur c et sa valeur mu
            nom_courbe = f"Sol {idx+1} : $c={c_trouve:.2f}$, $\mu={mu_trouve:.2f}$"
            # Superpose la courbe de cette solution spécifique sur le graphique en lui attribuant son nom
            tracer_solution_Q2(sol_x, sol_y, c_trouve, mu_trouve, label=nom_courbe)

        # Restreint la zone d'affichage de l'axe horizontal entre les valeurs -0.5 et 1
        plt.xlim(-0.5, 1)
        plt.title("Comparaison de toutes les solutions $Q(x)$ trouvées", fontsize=14)
        plt.xlabel("$x$ (Espace)", fontsize=12)
        plt.ylabel("$Q(x)$", fontsize=12)
        plt.grid(True, alpha=0.4)
        plt.legend(loc="upper right")
        plt.tight_layout()
        
        # Déclenche l'affichage effectif de la figure complète combinée sur l'écran
        plt.show()
    else:
        print("Aucun paramètre valide trouvé pour g_oscillante.")


    # --- 1. CAS FONCTION G SUITE AVEC 2 MAXIMUM ---


    Longueur_x_P = 4.7 # Longueur x pour le tracé de P
    Longueur_x_g = 10 # Longueur x pour le tracé de g
    Longueur_x_solution = 100 # Longueur x pour le tracé de la solution Q
    x_max_recherche = 100 # Longueur pour rechercher mu et c
    nb_points_recherche = 3000 # nombre de points pour décomposer l'intervalle pour la recherche de mu et c
    nb_points = 20000 # nombre de points pour la solutions
    print("--- Analyse de g_suite ---")
    plt.figure()

    # Boucle de 1 à 9 inclus (avec un pas de 1) pour tracer les différentes fonctions de la suite
    for i in range(1, 10, 1) :
        # Superpose la courbe de P(x) correspondant à l'indice i actuel sur le premier graphique
        tracer_P_suite(g_suite, P_suite, i, Longueur_x_P)
        
    # Affiche la première figure contenant toutes les courbes de P(x) superposées
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.axhline(0, color="black", linestyle="--", linewidth=2)
    # Parcourt une liste spécifique de valeurs de n pour étudier le comportement asymptotique des solutions
    for n in [1, 10, 50, 300, 1000, 10000]:
        # Cherche les couples de paramètres (c, mu) valides pour la fonction g_suite au rang n donné
        parametres_valides = trouver_c_mu_g_suite(g_suite, n, x_max=x_max_recherche, nb_points=nb_points_recherche)
        
        # Extrait le deuxième couple de paramètres trouvé (indice 1) de la liste des résultats
        c_trouve, mu_trouve = parametres_valides[1]
        # Calcule numériquement les coordonnées x et y de la solution Q(x) pour ce rang n et ces paramètres précis
        sol_x, sol_y = trouver_solution_Q_g_suite(mu_trouve, c_trouve, g_suite, n, Longueur_x=Longueur_x_solution, n_points = nb_points)
        # Construit l'étiquette de texte dynamique qui sera affichée dans la légende pour cette courbe
        nom_courbe = f"Sol pour n={n} : $c={c_trouve:.2f}$, $\mu={mu_trouve:.5f}$"
        # Superpose la courbe de cette solution spécifique sur le second graphique comparatif
        tracer_solution_Q2(sol_x, sol_y, c_trouve, mu_trouve, label=nom_courbe)

    # Restreint la zone de visualisation de l'axe horizontal entre les valeurs -0.5 et 8
    plt.xlim(-0.5, 8)
    plt.title("Comparaison des solutions $Q(x)$ pour plusieurs n", fontsize=14)
    plt.xlabel("$x$ (Espace)", fontsize=12)
    plt.ylabel("$Q(x)$", fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()
    # Déclenche l'affichage final de la figure à l'écran
    plt.show()