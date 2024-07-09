# GITHUB REPO URL

https://github.com/9MNickHazard/mygitrepo

# PROBLEM AREA/AREA OF INTEREST

The problem statement that I decided to go for won't change the world, but it is a small drop in a large pool of statistical analysis and machine learning that is very much on the rise right now in the world of eSports. The eSports organization that I co-founded, 9Moons, runs weekly tournaments for both Fighting Games (such as Street Fighter, Tekken, Guilty Gear and more) as well as Rocket League, which is basically soccer but played with cars that can fly and boost and spin, and is a very technically demanding and difficult game. My original project for this Capstone Project was to use player's records as well as some secondary variables to predict who would win between two competitors playing a fighting game. After having fun with the statistics part of that, I realized that there simply weren't enough highly correlated secondary variables (such as location, characters, time of year, etc.) to warrant the use of a ML model over some much simpler method of determining a win, like the ELO system (widely used in the chess world, among other games/sports). After this realization, I decided to switch to Rocket League data because I know that there is much more tertiary data that is obtainable directly from the game itself that could be more indicative of skill level and therefore win probability. To my knowledge, there is no one out there that has trained a ML model on these tertiary stats and so, armed with many more variables I could use to predict a win, I set off on this new course. The problem statement is to use tertiary stats (this simply means not including things like shots/saves/goals/assists/etc.) of each player on each team in a 3v3 format in Rocket League to predict the outcome of the game.

SPRINT 3 UPDATE: 

Not much to update here, as the problem area has stayed the same.





# SOLUTION

First I had to obtain a bunch of random RL (Rocket League) games from a 3rd party website, called ballchasing.com, that hosts servers that store tons of stats from games uploaded by players. I had to re-learn how to write code to pull from an API, and did quite a bit of research and documentation reading (from ballchasing.com's API documentation) to get it working, but eventually I was able to pull the last 50,000 games with all the stats (there are something like 88 different stats for each player in each game). This was quite the process, as it took multiple 6+ hour attempts of running the queries, encountering errors and having to run it again, etc. until I was able to get all 50,000 games. After loading the data from the json file query output into a pandas dataframe (which was about 300k lines, 6 rows, 1 for each player, per game as each game was 3v3), there are many stats that would obviously not be highly correlated with a win, such as "amount of boost collected overfill" (which just means once you have 100 boost, if you collect any more, it's wasted, as 100 is the max amount you can have at once), that I took out immediately. After some EDA and feature engineering, I ended up with about 20 tertiary stats, out of the original 88, that were correlated enough that I could proceed to modeling. I first tried a Neural Network, but I don't believe I had enough data (or really the knowledge of the intricacies of how NN's work) to make the model better than flipping a coin. After a period of frustration, I settled on Logistic regression to predict a binary outcome of win/loss, based on the features I had chosen. I flattened the 3 rows for each team into a 1D array, as well as some other preprocessing and fed it all into a logistic regression model. The outcome was good, at 75% accuracy on predictions, but I wanted it a bit better. After some more feature engineering and a few other tweaks, I got it up to 81%, where it sits today. In conclusion, I am using a select number of more highly correlated tertiary stats from a stat aggregating website that I pulled the data from myself to predict win/loss of a RL game given the lifetime average stats of all players in the game, via a logistic regression model (as well as SVM and XGBoost, although Logistic Regression proved to be the fastest to train while still being tied for the most accurate).

SPRINT 3 UPDATE:

I queried Ballchasing again for 50,000 more games, bringing the total to 100,000 games. Added some important Visualizations to get a better understanding of the data. The Correlation Heatmap was especially helpful to see highly positively/negatively correlated features and this was used extensively to alter the feature columns used for training. I also expanded the models I used to 5: Logistic Regression, Random Forest, XGBoost, LightGBM and CatBoost. I also altered the prepare_game_data function because I realized that I had not really been predicting the proper thing in Sprint 2. It was combinging the 6 rows that are in one game into one row but not properly adding the winner column in a fashion that reflected a 1 as blue team win and a 0 as orange team win. In the end, I'm not sure if it mattered though, as the results are pretty much the same as before and Im 100% sure that the function is setting up the training/test data properly now. I think the lack of improvement in modeling results, despite the advanced modeling, feature engineering and double the data size, is largely due to the fact that it's hard to predict games with these tertiary statistics. In Rocket League, a team could be way more skilled, hence having much higher stats in the highly win-correlated features, and still lose to a worse team because something went wrong, or the other team got a few lucky goals, etc. It's simply not a fool-proof way of predicting wins, and for that reason, I believe 81-82% accuracy in win prediction is actually quite good. There may be something to be said about training a more complex Neural Network on 10 or 100x the data I have, but that would be something outside the scope of this project as it would likely take weeks and weeks to query an API for that much data (given rate limits) and then train a NN on my desktop, despite my desktop being quite high-end. It already takes hours for Random Forest to train on a dataset that is 100,000 rows and 78 columns, a day or so for SVM, and NN even longer. I think at the end of the day, for this particular problem, the speed and relative accuracy that a simple Logistic Regression model gives is unmatched. As for next steps in this projects, if I had unlimited resources and time I would definitely try a NN on a truly massive dataset, but my guess is that even that would have a hard time getting above 85% accuracy. I would also maybe include main stats, such as average lifetime goals/assists/saves, but that would take quite a bit of alteration in the modeling processes. I plan to bring my models to 9Moons commentary to give our casters another interesting thing to use to predict games and have something to talk about before our weekly tournaments.





# IMPACT OF SOLUTION

I am obviously not doing something like predicting cancer or movie recommendations here, but I wanted to make sure that my Capstone Project was done on something that truly interested me, even if that meant that the impact of the final solution was a bit less impactful than some other "more serious" projects. This model could certainly be used to do eSports betting on large RL events, it could also help my own organization by seeing if fair teams have been created for our Rocket League Season Events that we run (fair teams is a big deal to the people in our community), or it could simply be used as fun statistics and predictions for RL eSports Casters/Commentators to talk about before a game. Although these impacts my not be world-changing, things like this are still needed in eSports to help develop the area and push progress forward.

SPRINT 3 UPDATE:

I think we have learned that there is an upper limit to how well a game can be predicted on tertiary stats. Although these are a decent indicator of skill, they certainly cannot always predict the outcomes. The impact of my solution remains the same, as a fun tool for commentators to discuss before a match or possibly for betting on games. Again, this was simply a fun project and area of interest for me as it related to 9Moons and my own interests, but is not a world changing solution.


# DATA DICTIONARY

Rocket League is basically Soccer, but played in a 3 vs 3 format with cars that can boost (speed up, using the boost resource, of which you can have 100 max at any given time; boost is also collected on the field as small boost pads which give 13 boost, and large boost pads which give 100 boost), jump (hitting the A button on the controller), flip (hitting the A button and a direction on the control stick while the car is already in the air), and fly (boost while your car is pointed upwards in the air). The point of the game is exactly the same as Soccer, with the team with the most goals at the end of 5 minutes winning the game. The field is also basically the same as a soccer field, being rectangular, with a goal at each end, but also walls and a ceiling, which can be driven on.

*NOTE: each of these stats is unique to one player on the team (of 3)*

- bpm - float64 - boost used per minute
- bcpm - float64 - boost collected per minute 
- amount_collected - float64 - total boost collected over the whole game
- amount_stolen - float64 - total boost collected on the opponents half of the field over the whole game
- percent_zero_boost - float64 - percent of total game time where the player has 0 boost in their boost gauge (aka not able to boost at all)
- percent_full_boost - float64 - percent of total game time where the player has maximum (100) boost in their boost gauge
- avg_speed - float64 - average speed (measured in some in-game engine units/second) over the whole game
- time_powerslide - float64 - amount of time spent powersliding (powersliding is simply turning while pressing the powerslide button, that lets you drift and turn faster at the cost of speed)
- count_powerslide - float64 - number of powerslides in the whole game (powerslide count is typically quite correlated with a higher level of skill; i.e. more powerslides (up to an extent) means the player is of higher skill level)
- percent_slow_speed - float64 - percent of total game time spent going at "slow speed" which is simply under a certain threshold of speed
- percent_boost_speed - float64 - percent of total game time spent going at "boost speed" which is simply over a certain threshold of speed
- percent_supersonic_speed - float64 - percent of total game time spent going at "supersonic speed" which is simply over a certain threshold of speed (higher than that of boost speed, this is the max speed in the game)
- percent_ground - float64 - percent of total game time spent on the ground
- percent_low_air - float64 - percent of total game time spent in the air but below 50% of the max height of the arena
- percent_high_air - float64 - percent of total game time spent in the air but above 50% of the max height of the arena
- avg_distance_to_ball_possession - float64 - average distance to the ball when your own team has ball possession (distance is again measured in in-game units)
- avg_distance_to_ball_no_possession - float64 - average distance to the ball when the opposing team has ball possession
- avg_distance_to_mates - float64 - average distance to your two teammates
- percent_defensive_third - float64 - percent of total game time spent in your defensive third of the field
- percent_offensive_third - float64 - percent of total game time spent in the offensive third of the field (aka your opponents defensive third)
- percent_neutral_third - float64 - percent of total game time spent in the middle third of the field
- percent_behind_ball - float64 - percent of total game time spend between your own goal and the ball
- percent_infront_ball - float64 - percent of total game time spend between your opponents goal and the ball
- win - bool - Value we are trying to predict. True for a win, False for a loss

SPRINT 3 UPDATE:

The final feature columns that I ended up with after removing more highly correlated features was this:
['bpm', 'bcpm', 'amount_stolen', 'percent_zero_boost', 'percent_full_boost', 'avg_speed', 'percent_boost_speed', 'percent_high_air', 'avg_distance_to_mates', 'percent_offensive_third', 'percent_neutral_third', 'percent_behind_ball', 'avg_distance_to_ball_possession']

These are all also listed above. Obvious removals were things like percent_defensive_third since that value is implied by the existence of percent_offensive_third and percent_neutral_third. There were a few instances of this. Although no matter how I altered the feature columns, I was not able to get above around 82% model accuracy, I found that these features provided the best average accuracy and the least training time. Lowering the number of features from here generally resulted in less than 80% accuracies.


# PROJECT ORGANIZATION

This project is split into two folders, Capstone_Project & API_Queries. The Capstone_Project folder contains the README file, as well as the EDA/Modeling Jupyter Notebook and the Presentation Slides. The API_Queries folder contains the Jupyter Notebook where I performed all my ballchasing.com API Queries.
