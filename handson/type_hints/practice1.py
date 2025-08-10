def get_full_name(first_name, last_name):
    full_name = first_name.title() + " " + last_name.title()
    return full_name

print(get_full_name("john", "doe"))


def get_full_name(first_name: str, last_name: str):

    full_name = first_name.title() + " " + last_name.title()
    return full_name

print(get_full_name("shank", "tmg"))



""" Above the two functions does the same, one thing is that the first one doesn't have no type hints or 
type annotations which doesn't supports python strings auto complete for methods. But the second function laced with 
type hints supports that.

"type hints": are special syntax that allow declaring the type of a variable
- declarign types for variables, editors and tools can give you better support.

"""