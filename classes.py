# Defining the class (template for Instagram users)
class InstagramUser:
    # creating the constructor method to initialize the attributes
    def __init__(self, username, bio, profile_picture):
        #Atrritbutes of the class: the properties that each Instagram user will have
        self.username = username
        self.bio = bio
        self.profile_picture = profile_picture

    # Methods of the class: the actions that each Instagram user can perform
    def follow(self, other_user):
        """Simulate following another user."""
        print(f"{self.username} followed {other_user.username}")
        # return f"{self.username} followed {other_user.username}"

# Creating instances (objects) of the InstagramUser class
user1 = InstagramUser("vin_andris", "Love coding and dancing.", "vin_profile.jpg")
user2 = InstagramUser("nashipae_waweru", "Adventure seeker and foodie.", "nashipae_profile.jpg")
user3 = InstagramUser("magymbugua", "Loves photography and travel.", "magy_profile.jpg")
user4 = InstagramUser("chrisbrownofficial", "King of RnB.", "breezy_profile.jpg") 
user5 = InstagramUser("wainaina_ndungu", "Loves hiking and flying.", "wainaina_profile.jpg")
user6 = InstagramUser("urvi_patel", "Adventure seeker and travel.", "urvi_profile.jpg")

# Accessing the attributes of the instances
print(user1.username)  # Output: vin_andris
print(user2.bio)       # Output: Adventure seeker and foodie.
print(user3.profile_picture)  # Output: magy_profile.jpg


# Simulating actions using the methods of the instances
user1.follow(user2)  # Output: vin_andris followed nashipae_waweru
user3.follow(user1)  # Output: magymbugua followed vin_andris
user3.follow(user4)  # Output: magymbugua followed chrisbrownofficial  