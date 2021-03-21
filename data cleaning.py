import pandas as pd
from datetime import datetime
import re


def timenow():
    return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ':')


print(timenow(), 'reading train.csv...')
df_train = pd.read_csv('train.csv')
df_train2 = df_train.copy()
print(timenow(), 'lowering case...')
df_train2['raw_address'] = df_train2.apply(lambda x: x['raw_address'].lower(), axis=1)
df_train2['POI/street'] = df_train2.apply(lambda x: x['POI/street'].lower(), axis=1)
nochange = pd.DataFrame()
abbrev = pd.DataFrame()
ambiguous = pd.DataFrame()


def clean_str(s):
    s = re.sub(r'[^a-zA-Z0-9,\.\-\s:()\/\"\'&#;]', '', s)
    whitelist = ",-:()/\"\'&#;"
    for i in whitelist:
        s = str(s).replace(i, f' {i} ')
    return s


def charsClean(input):
    whitelist = 'qwertyuiopasdfghjklzxcvbnm(%<.#"@;!+-]/=|&_)` ?~[,:\''
    # Replace all meaningless symbols with space
    cleanStr = str(input).replace('%', ' ').replace('<', ' ').replace('"', ' ').replace(';', ' ').replace('!',
                                                                                                          ' ').replace(
        '+', ' ').replace(']', ' ').replace('=', ' ').replace('|', ' ').replace('&', ' ').replace('_', ' ').replace('`',
                                                                                                                    ' ').replace(
        '?', ' ').replace('~', ' ').replace('[', ' ').replace('\'', ' ').replace('..', '. ')
    words = str(cleanStr).strip().split()
    cleanAddress, cleanWords = [], []
    for word in words:
        cleanWords += [''.join([i for i in word if i in whitelist])]

    cleanAddress = ' '.join(cleanWords)
    return cleanAddress


print(timenow(), 'Processing punctuations...')


# df_train2['raw_address'] = df_train2.apply(lambda x: charsClean(x['raw_address']), axis=1)
# df_train2['POI/street'] = df_train2.apply(lambda x: charsClean(x['POI/street']), axis=1)
# df_train2['raw_address'] = df_train2.apply(lambda x: clean_str(x['raw_address']), axis=1)
# df_train2['POI/street'] = df_train2.apply(lambda x: clean_str(x['POI/street']), axis=1)


def abbreviate(id, a, p):
    global nochange, abbrev, ambiguous
    address = str(a).split()  # do your whatever regex cleaning here or above
    POI, street = str(p).split('/')
    POI = POI.split()
    street = street.split()
    address_f = [i[0] for i in address]  # all the first letters
    POI_f = [p[0] for p in POI]
    street_f = [s[0] for s in street]
    len_a = len(address_f)
    len_p = len(POI_f)
    len_s = len(street_f)
    p_point, s_point = [], []
    returnOri = False
    for i in range(
            len_a - len_p + 1):  # bcuz the match is continuous so the last possible location of the letter is lena - lenp
        if address_f[i:(len_p + i)] == POI_f and POI_f != []:
            p_point.append(i)  # append this point i to a list (in case got multiple) if this whole block has same first letter as the original
        if address_f[i:(len_s + i)] == street_f and street_f != []:
            s_point.append(i)
    # print(address_f, POI_f, street_f, p_point, s_point)

    clean_a = ' '.join(address)
    if not all(x in address for x in POI):
        if len(p_point) == 1:
            # print(address[p_point[0]:(len_p + p_point[0])])
            if address[p_point[0]:(len_p + p_point[0])] != POI:
                for _ in range(p_point[0], (len_p + p_point[0])):
                    address.insert(_, POI[_ - p_point[0]])
                    address.pop(_ + 1)
        elif len(p_point) >= 1:
            returnOri = True
            pass
            # return str(a)
            # temp = pd.DataFrame([[id, str(a), str(p)]])
            # print(temp)
            # ambiguous = pd.concat([ambiguous, temp], ignore_index=True)
    if not all(x in address for x in street):
        if len(s_point) == 1:
            # print(address[s_point[0]:(len_s + s_point[0])])
            if address[s_point[0]:(len_s + s_point[0])] != street:
                for _ in range(s_point[0], (len_s + s_point[0])):
                    address.insert(_, street[_ - s_point[0]])
                    address.pop(_ + 1)
        elif len(s_point) >= 1:
            returnOri = True
            pass
            # return str(a)
            # temp = pd.DataFrame([[id, str(a), str(p)]])
            # print(temp)
            # ambiguous = pd.concat([ambiguous, temp], ignore_index=True)
    updated_a = ' '.join(address)
    if clean_a == updated_a:
        pass
        # return updated_a
        # temp = pd.DataFrame([[id, updated_a, str(p)]])
        # return temp['1']
        # print(temp)
        # nochange = pd.concat([nochange, temp], ignore_index=True)
    else:
        temp = pd.DataFrame([[id, str(a), updated_a]])
        # print(temp)
        abbrev = pd.concat([abbrev, temp], ignore_index=True)

    if returnOri:
        return str(a)
    else:
        updated_a = ' '.join(address)
        return updated_a
    # print(clean_a, updated_a)


print(timenow(), 'Expanding short forms...')
df_train2['raw_address'] = df_train2.apply(lambda x: abbreviate(x['id'], x['raw_address'], x['POI/street']), axis=1)
df_train2.to_csv('train_clean_expanded.csv', index=False)
abbrev.to_csv('abbrev.csv')
print(timenow(), 'Expanded data saved.')
#
# for k in range(100):  # CHANGE THIS TO RUN HOW MANY TIMES
#     num = df_train2.iat[k, 0]
#     data = df_train2.iat[k, 1]
#     label = df_train2.iat[k, 2]
#     abbreviate(num, data, label)

# print('no change\n', nochange)
# print('abbreviated\n', abbrev)
# print('ambiguous\n', ambiguous)

print()
