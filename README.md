### Introduction
This Recommender uses FunkSVD to make predictions of exact ratings for certrain user-movie pairs. And uses either FunkSVD or a Knowledge Based recommendation (highest ranked) to make recommendations for users. Finally, if given a movie, the recommender will provide movies that are most similar as a Content Based Recommender.

### Quickstart
Setup the class </br>
`rec = r.Recommender()` </br>

Provide the file path and fitting parameters </br>
`rec.fit(reviews_pth='train_data.csv', movies_pth= 'movies_clean.csv', learning_rate=.01, iters=5)` </br>

Make predictions of rating for user-movie pair</br>
`rec.predict_rating(user_id=8, movie_id=2844)` </br>

Make the recommendation for certain user (user_id) or movie (movie_id) </br>
`rec.make_recs(8, _id_type='user', rec_num=10)`

### Requirments
`python3`
`numpy`
`pandas`

### Acknowlegement
This class is adapted from udacity practice.
