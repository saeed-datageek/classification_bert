import re

emails = '''
CoreyMSchafer@gmail.com
corey.schafer@university.edu
corey-321-schafer@my-work.net
'''

pattern = re.compile(r'[a-zA-Z0-9.-]+@[a-zA-Z-]+\.(com|net)')
matches = pattern.finditer(emails)
print('test')
for matche in matches:
    print(matche)
