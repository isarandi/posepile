import json


def main():
    with open('takes.json') as f:
        data = json.load(f)
        for item in data:
            print(item['take_uid'], item['take_name'])


if __name__ == '__main__':
    main()
