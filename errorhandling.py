def validate_and_execute(days):
    try:
        user_input_number = int(days)
        if user_input_number > 0 and user_input_number <= 7:
            calculated_value = (user_input_number/ 7)* 100
            print(f"Your workout attendance is {calculated_value:.1f}%")
        elif user_input_number < 0:
            print("Your entered a number below 0, please ennter a valid positive number")

        elif user_input_number > 7:
            print("Number of days cannot be more than 7")

    except ValueError:
        print("Your inout is not a valid number. Don't ruin my program!")

user_input = input("Hey user, enter the number of days that you work out per week? ")  
validate_and_execute(user_input)
