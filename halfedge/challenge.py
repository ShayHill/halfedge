from urllib import request
import re


def get_source_comment(puzzle_name: str) -> str:
    """Extract comment block from a Python Challenge puzzle page source.

    Puzzle materials are contained at www.pythonchallence.com/pc/def/:puzzle_name:
    inside html comment tags "<!--" => "-->"

    1. attempt to read comment from cache
    2. attempt to read from Net. Create cache
    3. fail with HTTPError
    """
    try:
        return open(f"puzzle_comments/{puzzle_name}.txt").read()

    except FileNotFoundError:
        url = f"http://www.pythonchallenge.com/pc/def/{puzzle_name}.html"
        page_source = request.urlopen(url).read().decode()
        comment = re.findall("<!--(.*?)-->", page_source, re.DOTALL)[-1]
        with open(f"puzzle_comments/{puzzle_name}.txt", "a") as cache_file:
            cache_file.write(comment)
        return comment


def problem_1() -> None:
    """Problem 1: map"""
    from string import ascii_lowercase

    coded_message = (
        "g fmnc wms bgblr rpylqjyrc gr zw fylb. rfyrq ufyr amknsrcpq ypc dmp."
        " bmgle gr gl zw fylb gq glcddgagclr ylb rfyr'q ufw rfgq rcvr gq qm jmle."
        " sqgle qrpgle.kyicrpylq() gq pcamkkclbcb. lmu ynnjw ml rfc spj."
    )

    trans_table = str.maketrans(
        ascii_lowercase, ascii_lowercase[2:] + ascii_lowercase[:2]
    )
    # print("problem 1: map -> " + coded_message.translate(trans_table))
    print("problem 1: map -> " + "map".translate(trans_table))


def problem_2() -> None:
    """Problem 2: ocr"""
    from collections import Counter

    mess = get_source_comment("ocr")
    rare_characters = [key for key, val in Counter(mess).items() if val == 1]
    print("problem 2: ocr -> " + "".join(rare_characters))


def problem_3() -> None:
    """Problem 3: equality"""
    mess = get_source_comment("equality")
    small_letters = re.findall("[^A-Z][A-Z]{3}([a-z])[A-Z]{3}[^A-Z]", mess)
    print("problem 3: equality -> " + "".join(small_letters))


def problem_4() -> None:
    """Problem 4: linkedlist

    Problem 4 is slow and only works online.
    """
    page_root = "http://www.pythonchallenge.com/pc/def/linkedlist.php?nothing="

    def find_next_instruction(next_nothing: str) -> str:
        """Get the next_nothing from the page at this_nothing.

        Most pages at page_root + next_nothing have a line containing 'next nothing
        is \d*'. Try to append that to page_root and keep going. If that doesn't
        work, the page contains a special instruction. Return that instruction.
        (instruction, next_nothing)
        """
        for i in range(400):
            page_source = request.urlopen(page_root + next_nothing).read().decode()
            try:
                next_nothing = re.search("next nothing is (\d*)", page_source).group(1)

            except AttributeError:
                return page_source, next_nothing
                break

    # do_next = find_next_instruction("12345") # take half of 16044
    do_next = find_next_instruction("8022")
    print("problem 4: linkedlist -> " + do_next[0])


problem_1()
problem_2()
problem_3()
# problem_4()
