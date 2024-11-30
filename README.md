# PreSnapPrediction

## Reception Zones and the ORPSP Metric
### Introduction
NFL pass playbooks are built around precise receiver routes and strategically chosen reception zones to exploit defensive gaps. Route combinations are designed to create space and exploit defensive weaknesses by coordinating multiple receivers' movements. Concepts like flooding a zone attack different depths in one area, while rub routes disrupt man-to-man coverage. Combinations such as post and corner stretch defenses vertically, while stick and flat create horizontal separation. These designs ensure at least one receiver is likely to be open, providing clear options for the quarterback.

For each play design, understanding how likely each receiver is to be open is key to execution. Factors like route depth, timing, and combinations dictate which receiver has the best chance of being in a favorable position. By recognizing these probabilities before the snap, quarterbacks can prioritize high-percentage options and deliver the ball with confidence and precision.


### Reception Zone Identification for Every Route
Each route is designed to guide the receiver into a specific area on the field where they could be targeted by the quarterback. Identifying these reception zones is essential for understanding the likelihood of a receiver being open.

In the tracking data, it is straightforward to identify the reception zones of receiver routes that are targeted. However, the goal is to determine them for all routes run. The strategy of determining reception zones for groups of routes is an efficient method to identify the reception zones for all routes, leveraging the targeted routes within each group. There are numerous variations of distinctly different routes within the same "routeRan" categories, and these routes do not always appear to be consistently classified. Therefore, performing clustering is necessary to create more precise route groups.

Routes from the same categorie can go in different directions depending on whether the receiver is on the left or right side of the football. For example, an out route on the left side goes in the opposite direction of an out route on the right side but in the same direction as an in route. While the two out routes may be identical, they would not be grouped together without preprocessing. To address this, it is simpler to consider out and in routes as 90-degrees angle routes and standardize them to go in the same direction. Consequently, all routes are standardized.

<p align="center">
    <img src="reports/figures/route_standardization.png">
    Figure 1. Route standardization
</p>

With all routes standardized to the same direction, clustering can be applied using key features such as the coefficients of a quadratic approximation, positions at specific frames and their standard deviation over time. Affinity Propagation is a robust choice that delivers accurate clustering results. The algorithm autonomously identifies cluster centers and assigns data points to clusters by evaluating both the similarity between data points and their suitability as exemplars. Reception zones can be assigned to the resulting clusters based on the targeted routes within each cluster.

<p align="center">
    <img src="reports/figures/route_clustering.png">
    Figure 2. Route clustering and identified reception zones
</p>

The average route time are calculated for each cluster, providing more valuable information. For each route run, a reception zone and its route time are determined by identifying its corresponding cluster. Below is an example play showcasing the computed reception zones and route times.

<p align="center">
    <img src="reports/animations/animated_play_routes.gif">
    Figure 3. Example play with reception zones
</p>


### Introducing the ORPSP Metric: Open Receiver Pre Snap Probability

### Metric Analysis and Use Cases

### Conclusion
