**Comment, dans un espace sans couverture GPS et sans matériel spécifique, est-il possible de se 
localiser c’est-à-dire de trouver la salle où se trouve une personne dans un bâtiment entier ?**


Une réponse potentielle qui est le sujet de cette étude est une localisation par détection 
d’images dans le cadre d’une navigation piétonne grand public. 

Le principe est simple, à l’aide d’un appareil que tout le monde possède comme un smartphone il faut
concevoir une solution de localisation : il faut dans un premier temps prendre une photo de son 
entourage et via une mise en correspondance sur une base de données, choisir l’image la plus proche 
et en déduire la localisation.
Une contrainte est de se focaliser sur les solutions les moins gourmandes en
termes de calculs et donc de délai de réponse. 
L’état de l’art actuel propose deux solutions qui peuvent répondre à notre problématique :
- La mise en correspondance image par similarité de points SIFT (Scale invariant feature 
transform) : 
Les points SIFT sont des points d’une image dits remarquables, ceux sont les points qui 
possèdent un gradient tridimensionnel issu des pixels les rendant uniques et in extenso qui 
permettent de décrire une image en se focalisant uniquement sur un set de points SIFT. Les 
avantages de cette solution sont doubles : 
✓ En ne gardant que les points SIFT, le stockage et les calculs sont fortement 
diminués (face à une comparaison par pixel).
✓ Il n’est pas nécessaire d’avoir un entrainement (1 image par localisation suffit) et 
donc la création de la base de données est plus simple.
En contrepartie, vu la perte élevée d’information, les résultats peuvent s’avérer moins bons 
que pour d’autres méthodes.
- La mise en correspondance image par réseaux de neurones (Apprentissage Profond). Cette 
solution est en général la plus efficace mais elles nécessitent une base de données 
extrêmement complète (base de données non disponible pour notre projet). Une alternative 
est l’utilisation d’un algorithme dits One Shot Learning (OSL) qui permet de s’affranchir du 
large besoin en données.
