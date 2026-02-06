"""
Create class called book with title, author, and price.
Should have a method called apply_discount (discount percentage) and then 
calculate the new price after applying the discount.
"""

class Book:
    def __init__(self, title, author, price):
        self.title = title
        self.author = author
        self.price = price

    def apply_discount(self, discount_percentage):
        """Calculate the new price after applying the discount."""
        discount_amount = self.price * (discount_percentage / 100)
        new_price = self.price - discount_amount
        return new_price
    
# Creating instances (objects) of the Book class
book1 = Book("Giant Steps", "Anthony Robbins", 11.99)
book2 = Book("Animal Farm", "George Orwell", 12.49)
book3 = Book("Outliers", "Malcom Gladwell", 10.15)

# Accessing the attributes of the instances
print(book1.title)  # Output: Giant Steps
print(book2.author)  # Output: George Orwell
print(book3.price)   # Output: 10.15

# Applying discount using the method of the instances
new_price_book1 = book1.apply_discount(10)  # 10% discount
print(f"New price of '{book1.title}' after discount: ${new_price_book1:.2f}")  # Output: New price of 'Giant Steps' after discount: $10.79


    