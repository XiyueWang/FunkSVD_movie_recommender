import numpy as np
import pandas as pd
import recommender_functions as rf
import sys # can use sys to take command line arguments

class Recommender():
    '''
    This Recommender uses FunkSVD to make predictions of exact ratings.  And uses
    either FunkSVD or a Knowledge Based recommendation (highest ranked) to make
    recommendations for users.  Finally, if given a movie, the recommender will
    provide movies that are most similar as a Content Based Recommender.
    '''
    def __init__(self):
        '''
        '''



    def fit(self, movies_pth, reviews_pth, latent_features=10, learning_rate=0.0001, iters=100):
        '''
        Input:
        movies_pth - path to movies csv
        reviews_pth - path to csv with at least the four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations
        Output:
        NONE
        movies -  df stores movies data
        reviews - df stores reviews data
        user_mat - user * latent features metrics
        movie_mat - latent features * movies metrics
        n_users - the number of users (int)
        n_movies - the number of movies (int)
        num_ratings - the number of ratings made (int)
        '''
        # Read the movies  dataframe
        self.movies = pd.read_csv(movies_pth)
        # Transform the reviews dataframe
        self.reviews = pd.read_csv(reviews_pth)
        self.reviews = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.reviews = self.reviews.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
        self.ratings_mat = np.array(self.reviews)

        self.n_users = self.ratings_mat.shape[0]
        self.n_movies = self.ratings_mat.shape[1]
        self.num_ratings = np.count_nonzero(~np.isnan(self.ratings_mat))

        user_mat = np.random.rand(self.n_users, self.latent_features)
        movie_mat = np.random.rand(self.latent_features, self.n_movies)

        sse_accum = 0
        print("Optimizaiton Statistics")
        print("Iterations | Mean Squared Error ")
        for iteration in range(self.iters):

            old_sse = sse_accum
            sse_accum = 0

            for i in range(self.n_users):
                for j in range(self.n_movies):
                    if self.ratings_mat[i, j] > 0:
                        diff = self.ratings_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])
                        sse_accum += diff**2
                        for k in range(self.latent_features):
                            user_mat[i, k] += self.learning_rate * (2*diff*ovie_mat[k, j])
                            movie_mat[k, j] += self.learning_rate * (2*diff*user_mat[i, k])
        print("%d \t\t %f" % (iteration+1, sse_accum / self.num_ratings))

        self.user_mat = user_mat
        self.movie_mat = movie_mat


    def predict_rating(self, user_id, movie_id):
        '''
        Input:
        user_id: user id you want to make the prediction
        movie_id: movie id you wan to make the prediction
        Output:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        '''
        self.user_ids_series = np.array(self.ratings_mat.index)
        self.movie_ids_series = np.array(self.ratings_mat.columns)

        try:# User row and Movie Column
            user_row = np.where(self.user_ids_series == user_id)[0][0]
            movie_col = np.where(self.movie_ids_series == movie_id)[0][0]

            # Take dot product of that row and column in U and V to make prediction
            pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])

            movie_name = str(self.movies[self.movies['movie_id'] == movie_id]['movie']) [5:]
            movie_name = movie_name.replace('\nName: movie, dtype: object', '')
            print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(pred, 2), str(movie_name)))

            return pred

        except:
            print("I'm sorry, but a prediction cannot be made for this user-movie pair.\
            It looks like one of these items does not exist in our current database.")

            return None


    def make_recs(self,_id, _id_type='movie', rec_num=5):
        '''
        given a user id or a movie that an individual likes to make recommendations.
        Input:
        _id - the user/movie id you want to predect for
        _id_type - the id _id_type
        rec_num - how many recommendation you want to provide
        Output:
        rec_names - (array) a list or numpy array of recommended movies by name
        '''
        rec_ids = create_ranked_df(self.movies, self.ratings_mat)

        if _id_type == 'user':
            if _id in self.ratings_mat.index:
                ind = np.where(self.ratings_mat.index == _id)[0][0]
                preds = np.dot(self.user_mat[ind,:],self.movie_mat)
                rec_inds = np.argsort(preds)[0-rec_num:]
                rec_ids = self.ratings_mat.columns[rec_inds]
                rec_names = rf.get_movie_names(rec_ids)
            else:
                rec_names = rf.popular_recommendations(_id, rec_num, rec_ids)
        else:
            rec_ids = rf.find_similar_movies(_id)
            rec_names = rf.get_movie_names(rec_ids)

        return rec_names

if __name__ == '__main__':
    import recommender as r

    rec = r.Recommender()

    rec.fit(reviews_pth='train_data.csv', movies_pth= 'movies_clean.csv', learning_rate=.01, iters=5)

    # predict
    rec.predict_rating(user_id=8, movie_id=2844)
