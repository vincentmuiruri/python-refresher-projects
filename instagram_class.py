class Instagram:
    def __init__(self, username, full_name, bio, followers_count, following_count, posts_count):
        self.username = username
        self.full_name = full_name
        self.bio = bio
        self.followers_count = followers_count
        self.following_count = following_count
        self.posts_count = posts_count

    def follow(self, other_user):
        """Simulate following another user."""
        other_user.followers_count += 1
        self.following_count += 1

    def unfollow(self, other_user):
        """Simulate unfollowing another user."""
        if other_user.followers_count > 0:
            other_user.followers_count -= 1
        if self.following_count > 0:
            self.following_count -= 1

    def post_photo(self):
        """Simulate posting a photo."""
        self.posts_count += 1

    def get_profile_info(self):
        """Return a summary of the user's profile information."""
        return {
            "Username": self.username,
            "Full Name": self.full_name,
            "Bio": self.bio,
            "Followers": self.followers_count,
            "Following": self.following_count,
            "Posts": self.posts_count
        }
    
# Creating instances (objects) of the InstagramUser class
user1 = Instagram("vin_andris", "Vin Andris", "Love photography and travel.", 150, 200, 35)
user2 = Instagram("nashipae_waweru", "Nashipae Waweru", "Foodie and adventure seeker.", 300, 180, 50)
# Accessing the attributes of the instances
print(user1.username)  # Output: john_doe
print(user2.bio)       # Output: Foodie and adventure seeker.
# Simulating actions using the methods of the instances
user1.follow(user2)
print(f"{user1.username} now has {user1.following_count} following.")
user2.unfollow(user1)
print(f"{user2.username} now has {user2.followers_count} followers.")
user1.post_photo()
print(f"{user1.username} now has {user1.posts_count} posts.")

# Getting profile information for User1
print("*"*60)
print("Vincent")
profile_info = user1.get_profile_info()
for key, value in profile_info.items():
    print(f"{key}: {value}")  

# Getting profile information for User1   
print("*"*60)
print("Nashipae")
profile_info = user2.get_profile_info()
for key, value in profile_info.items():
    print(f"{key}: {value}")
