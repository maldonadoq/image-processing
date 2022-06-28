# enconding
def encoding(data):
    # Building and initializing the dictionary.
    dictionary_size = 3
    dictionary = dict({
        'a': 0,
        'b': 1,
        'c': 2,
    })

    # We'll start off our phrase as empty and add characters to it as we encounter them
    phrase = ""

    # This will store the sequence of codes we'll eventually write to disk
    output = []

    # Iterating through the input text character by character
    print('{:5s}{:10s}{:4s}{:4s}'.format('Str', 'Out Code', 'Add', 'Code'))
    for symbol in data:

        # Get input symbol.
        string_plus_symbol = phrase + symbol

        # If we have a match, we'll skip over it
        # This is how we build up to support larger phrases
        if string_plus_symbol in dictionary:
            phrase = string_plus_symbol
        else:
            print('{:4s}{:4d}       {:4s}{:4d}'.format(
                phrase, dictionary[phrase], string_plus_symbol, dictionary_size))
            # We'll add the existing phrase (without the breaking character) to our output
            output.append(dictionary[phrase])

            # We'll create a new code (if space permits)

            dictionary[string_plus_symbol] = dictionary_size
            dictionary_size += 1
            phrase = symbol

    if phrase in dictionary:
        output.append(dictionary[phrase])

    return output


if __name__ == "__main__":

    word = 'aaaabbbccc'

    encode = encoding(word)
    #decode = decoding(encode)

    snorm = len(word)

    print('Initial     : ', word)
    print('Encode      : ', encode)
