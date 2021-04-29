import random


class Person_Manager():
    '''
    This class contains information about feature space and categories of the users
    '''

    def __init__(self, categories, probabilities, features):
        '''
        receives in input the users categories, candidate probabilities and the user's features
        '''
        self.categories = categories
        self.n_categories = len(self.categories)
        self.probabilities = probabilities
        self.features = features

        self.persons_count = 0
        # empty list of 4 categories (2 x 2 features)
        self.categories_count = [0] * self.n_categories

    def new_person(self):
        '''
        generates a new user with a given category with uniform probability
        '''
        p_category = random.randint(0, self.n_categories - 1)  # [0,1,2,3]
        self.persons_count += 1
        self.categories_count[p_category] += 1
        return p_category