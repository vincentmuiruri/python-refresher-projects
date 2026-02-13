#loading the libraries
from getpass import getpass
import hashlib

#simulatige a user database with hashed passwords
user_database = {
    "vin_andris": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # password: "password123"
    "magymbugua": "6ca13d52ca70c883e0f0bb101e425a89e8624de51db2d2392593af6a84118090"     # password: "securepass"
}

def hash_password(password):
    """Hash the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password): 
    """
- Checks if username exists in the user database
- Hashes the input password
- Compares hashed input with stored hash
- Returns Authentication message if successful or not
"""
    if username in user_database:
        hashed_input_password = hash_password(password)
        if hashed_input_password == user_database[username]:
            print("Authentication successful! Welcome back, " + username + "!")
        else:
            print("Authentication failed! Incorrect password.")
    else:
        print("Authentication failed! Username not found.")

# main section
def main():
    """Main function to run the login system.
    """
    print("=== Welcome to the Simple Login System ===")
    attempts = 3

    while attempts > 0:
        username = input("Enter your username: ")
        password = getpass("Enter your password: ")

        if authenticate_user(username, password):
            print(f"\n✓ Login successful! Welcome, {username}!")
            print("\n--- Accessing your dashboard ---")
            print("1. View Profile")
            print("2. Settings")
            print("3. Logout")
            break

        else:
            attempts -= 1
            if attempts > 0:
                print(f"Invalid credentials. You have {attempts} attempts left.")
            else:
                print("Too many failed attempts. Access denied.")

if __name__ == "__main__":
    main()


