# Authorization Preprocessing

import os
from io import BytesIO, StringIO
import boto3
from time import time
import numpy as np
import pandas as pd
from datetime import datetime
from awsglue.utils import getResolvedOptions
import sys
import gzip
from copy import deepcopy

# Start Time counter
start_time = time()

# Input and Output Buckets
args = getResolvedOptions(sys.argv, ['source_bucket', 'tableau_bucket'])

bucket = args["source_bucket"]
tableau_bucket = args["tableau_bucket"]

# File paths
# ag_key = 'ACCOUNT_GRIP_TRANSPARENCY/ACCOUNT_GRIP_TRANSPARENCY.csv.gz'
ag_key = 'T_CUSTOMER_FACTSHEET/CFS_ACCOUNT_GRIP_TRANSPARENCY/CFS_ACCOUNT_GRIP_TRANSPARENCY.csv.gz'
rgm_key = 'T_CUSTOMER_FACTSHEET/T_USER_ROLE/T_USER_ROLE.csv.gz'
lcm_key = 'T_CUSTOMER_FACTSHEET/T_TERRITORY_STRUCTURE/T_TERRITORY_STRUCTURE.csv.gz'
sch_key = 'temp/SieSales Champions.csv'
schi_key = 'temp/List of Accounts for CFS w IFA.csv'
ccm_key = 'temp/country_code_mapping.csv'
cuc_key = 'temp/country-codes_csv.csv'
curr_key = 'T_CUSTOMER_FACTSHEET/T_TCURR/CURRENCY_TCURR.csv.gz'
au_key = 'temp/add_user.csv'


# Define functions
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem)) 


def clean_conflicting_chars(df):
    df.columns = [x.replace(' ','_') for x in df.columns]
    df.columns = [x.replace('-','_') for x in df.columns]
    df.columns = [x.upper() for x in df.columns]
    for i in df.columns:
        try:
            df[i] = df[i].str.replace(';',' ')
            df[i] = df[i].str.replace('"',' ')
            df[i] = df[i].str.replace('\t',' ')
            df[i] = df[i].str.replace('\n',' ')
        except:
            continue
    reduce_mem_usage(df)
    return df


def remove_specific_user_roles(dframe, string):
    print(dframe.USER_ROLE.str.contains(string).sum(), '(', round(authorization.USER_ROLE.str.contains("CN").sum()/len(authorization)*100,3),'% ) IFA-GID combinations of', string, 'USER ROLES deleted.')
    dframe = dframe[~dframe['USER_ROLE'].str.contains(string)]
    
    return dframe


def delete_deprecated_files(bucket, prefix):
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    dates = dict()
    for object in response['Contents']:
        dates[object['Key']] = datetime(object['LastModified'].year, object['LastModified'].month, object['LastModified'].day)

    older_files_to_delete = [key for key in dates.keys() if dates[key] != max(dates.values())]

    for key in older_files_to_delete:
        print('Deleting', key)
        s3_client.delete_object(Bucket=bucket, Key=key)
    
        
# Account GRIP
print('\n ---- Read Account Grip From S3 & reduce memory usage ----')
ag_obj = boto3.client('s3').get_object(Bucket=bucket, Key=ag_key)
account_grip = pd.read_csv(BytesIO(ag_obj['Body'].read()),
                           sep=';',
                           quotechar='"',
                           dtype = 'object',
                           compression='gzip')
print(ag_key, 'last modified:', ag_obj['LastModified'])
reduce_mem_usage(account_grip)



account_grip.rename(columns={'ACCOUNT_ID':'ACCOUNTID',
                             'ACCOUNT_IFA':'IFA',
                             'User Role':'USER_ROLE'}, inplace=True)


print('\n ---- Drop null IFAs ----')
print(account_grip.IFA.isna().sum(), '(', round(account_grip.IFA.isna().sum()/len(account_grip)*100,3),'% ) IFAs are null. IFAs deleted.')
account_grip = account_grip.dropna(subset=['IFA']).reset_index(drop=True)
print('File contains', len(set(account_grip.IFA)), 'different IFAs and', len(set(account_grip.GID)), 'different GIDs')



# Introduce Global User
account_grip_global_user = account_grip.drop_duplicates(subset=account_grip.columns.difference(['GID', 'TRANSPARENCY_POSITION', 'USER_ID', 'COMPANY_ID', 'ACCOUNT_NAME'])).dropna()
account_grip_global_user['GID'] = 'Z0043ZVM'
account_grip_global_user.drop_duplicates(subset=['IFA','GID'],inplace=True)
account_grip = pd.concat([account_grip, account_grip_global_user])
print('Z0043ZVM' in list(account_grip['GID']))



# ROLE-GID
print('\n ---- Read GID - role mapping from S3 (temporary file) & reduce memory usage ----')
rgm_obj = boto3.client('s3').get_object(Bucket=bucket, Key=rgm_key)
role_gid_mapping = pd.read_csv(BytesIO(rgm_obj['Body'].read()),
                               sep=';',
                               quotechar='"',
                               dtype = 'object',
                               compression='gzip')
print(rgm_key, 'last modified:', rgm_obj['LastModified'])
reduce_mem_usage(role_gid_mapping)

role_gid_mapping['GID'] = role_gid_mapping['GID'].str.upper()
role_gid_mapping = role_gid_mapping[role_gid_mapping.GID.isna()==False]
role_gid_mapping['GID'] = role_gid_mapping['GID'].str[:8]
role_gid_mapping = role_gid_mapping.drop_duplicates(subset='GID')

role_gid_mapping['USER_ROLE'] = np.where(role_gid_mapping['GID'] == 'Z0043ZVM', 'FULL ACCESS',role_gid_mapping['USER_ROLE'])


print('\n ---- Save role_gid_mapping_transformed ----')
csv_buffer = StringIO()
role_gid_mapping.to_csv(csv_buffer, index_label='INDEX', sep=';', index=False)
gzip_object = gzip.compress(csv_buffer.getvalue().encode('utf-8'))
obj = boto3.resource('s3').Object(tableau_bucket, 'interm/role_gid_mapping.csv.gz')
obj.put(Body=gzip_object, ServerSideEncryption='aws:kms')
del role_gid_mapping['USER_ID']



# lead Country - Sales Country
print('\n ---- Read Lead Country - Sales Country mapping from S3 (temporary file) & reduce memory usage ----')
lc_data = boto3.client('s3').get_object(Bucket=bucket, Key=lcm_key)
lc_sc_mapping = pd.read_csv(BytesIO(lc_data['Body'].read()),
                            sep=';',
                            quotechar='"',
                            dtype = 'object', compression='gzip')
lc_sc_mapping = lc_sc_mapping[['S4SLEAD_COUNTRY__C','S4SSALE_COUNTRY__C']].drop_duplicates()
reduce_mem_usage(lc_sc_mapping)
lc_sc_mapping = lc_sc_mapping.rename(columns={'S4SSALE_COUNTRY__C':'Sales Country',
                                             'S4SLEAD_COUNTRY__C':'Lead Country'})
lc_sc_mapping = lc_sc_mapping.dropna(subset=['Sales Country'])
lc_sc_mapping = lc_sc_mapping.sort_values(by=['Lead Country','Sales Country']).reset_index(drop=True)


S = ', '.join(str(v) for v in set(lc_sc_mapping['Sales Country']))

lc_sc_mapping = lc_sc_mapping.reset_index()
lc_sc_mapping = clean_conflicting_chars(lc_sc_mapping)
del lc_sc_mapping['INDEX']


# Currency and Country Codes
cuc_obj = boto3.client('s3').get_object(Bucket=tableau_bucket, Key=cuc_key)
curr_codes = pd.read_csv(BytesIO(cuc_obj['Body'].read()), sep=',', quotechar='"', dtype = 'object')
reduce_mem_usage(curr_codes)

#Currency exchange table
print('\n ---- Read currency exchange ratings mapping ----')
curr_data = boto3.client('s3').get_object(Bucket=bucket, Key=curr_key)
curr = pd.read_csv(BytesIO(curr_data['Body'].read()), sep=';', compression='gzip')
reduce_mem_usage(curr)

# Currency and Country Codes
au_obj = boto3.client('s3').get_object(Bucket=tableau_bucket, Key=au_key)
add_user = pd.read_csv(BytesIO(au_obj['Body'].read()), sep=',', quotechar='"', dtype = 'object')
reduce_mem_usage(add_user)
add_user['IFA'] = add_user['IFA'].str.zfill(10)

# Currency code prepocessing
curr_codes = curr_codes[['ISO3166-1-Alpha-2','ISO4217-currency_name','ISO4217-currency_alphabetic_code', 'ISO4217-currency_country_name']].dropna().rename(columns={'ISO3166-1-Alpha-2':'COUNTRY_CODE','ISO4217-currency_name':'CURR_NAME','ISO4217-currency_alphabetic_code':'CURR', 'ISO4217-currency_country_name':'COUNTRY'})
special = curr_codes[curr_codes['CURR'].str.contains(',')]['COUNTRY_CODE']

special_usd = curr_codes[(curr_codes['CURR'].str.contains(','))&(curr_codes['CURR'].str.contains('USD'))]
special_usd['CURR_NAME'] = special_usd['CURR_NAME'].str.split(',').str[1]
special_usd['CURR'] = special_usd['CURR'].str.split(',').str[1]

rest = curr_codes[(curr_codes['CURR'].str.contains(','))&(~curr_codes['CURR'].str.contains('USD'))]
rest['CURR_NAME'] = rest['CURR_NAME'].str.split(',').str[0]
rest['CURR'] = rest['CURR'].str.split(',').str[0]

curr_codes = curr_codes[~curr_codes['COUNTRY_CODE'].isin(set(curr_codes[curr_codes['CURR'].str.contains(',')]['COUNTRY_CODE']))]

curr_codes = pd.concat([curr_codes, special_usd, rest])
curr_codes.head(10)

#IFA_CURRECY Table
print('\n ---- Map ifas with currency codes ----')
ifa_country = account_grip[['IFA','ACCOUNT_COUNTRY']].drop_duplicates()
ifa_country = ifa_country.merge(curr_codes[['COUNTRY_CODE','CURR']],left_on='ACCOUNT_COUNTRY', right_on='COUNTRY_CODE', how='left')
print('countries with currency problems \n', ifa_country[ifa_country['CURR'].isna()==True]['ACCOUNT_COUNTRY'].value_counts())
ifa_country['CURR'] = np.where(ifa_country['ACCOUNT_COUNTRY']=='TW','TWD', ifa_country['CURR'] )
ifa_country['CURR'] = ifa_country['CURR'].fillna('EUR')

print('\n ---- Save ifa_currency mapping----')
csv_buffer = StringIO()
ifa_country.to_csv(csv_buffer, index_label='INDEX', sep=';', index=False)
gzip_object = gzip.compress(csv_buffer.getvalue().encode('utf-8'))
obj = boto3.resource('s3').Object(tableau_bucket, 
                                  'interm/ifa_currency.csv.gz')
obj.put(Body=gzip_object, ServerSideEncryption='aws:kms')


#Processing Authorization
print('\n ---- Temporary merge with GID - User Role mapping & Renaming ----')
authorization = account_grip.merge(role_gid_mapping, on=['GID'], how='left')


print('\n ---- Drop nan, CN, HQ, GLO User Roles ----')
to_delete = authorization[(authorization['USER_ROLE'].str.contains('CN'))|\
                          (authorization['USER_ROLE'].str.contains('HQ'))|\
                          (authorization['USER_ROLE'].str.contains('GLO')|\
                           (authorization['USER_ROLE'].isna()))]
users_deleted_from_authorization = to_delete.groupby(by=['GID', 'USER_ROLE'])['IFA'].count().reset_index()

print(len(authorization[authorization['USER_ROLE'].isna()==True]['GID'].value_counts()), 'GIDs not found in GID-Role mapping')
print(authorization[authorization['USER_ROLE'].isna()==True]['GID'].value_counts())

print(authorization.USER_ROLE.isna().sum(), '(', round(authorization.USER_ROLE.isna().sum()/len(authorization)*100,3),'% ) USER ROLES are null. USER ROLES deleted.')
authorization = authorization.dropna(subset=['USER_ROLE']).reset_index(drop=True)
authorization = remove_specific_user_roles(authorization, 'CN')
authorization = remove_specific_user_roles(authorization, 'GLO')
authorization = remove_specific_user_roles(authorization, 'HQ')
print(len(users_deleted_from_authorization), 'Users deleted from Authorization because of their roles')

user_info = authorization[['GID', 'USER_ID', 'TRANSPARENCY_POSITION']].drop_duplicates()
ifa_info = authorization[['ACCOUNTID', 'IFA', 'COMPANY_ID', 'ACCOUNT_NAME', 'ACCOUNT_COUNTRY']].drop_duplicates()

# Load Full Access Users
print('\n ---- Read Sales Champions ----')
sch_obj = boto3.client('s3').get_object(Bucket=tableau_bucket, Key=sch_key)
champions = pd.read_csv(BytesIO(sch_obj['Body'].read()),
                           sep=',',
                           quotechar='"',
                           dtype = 'object')
reduce_mem_usage(champions)

print('\n ---- Read Champions ifas ----')
schi_obj = boto3.client('s3').get_object(Bucket=tableau_bucket, Key=schi_key)
champions_ifas = pd.read_csv(BytesIO(schi_obj['Body'].read()),
                           sep=',',
                           quotechar='"',
                           dtype = 'object')
reduce_mem_usage(champions_ifas)
champions_ifas = champions_ifas.rename(columns = {'IFA ':'IFA'})
champions_ifas['IFA'] = champions_ifas['IFA'].str.zfill(10)

print('\n ---- Read country_code_mapping From S3 & reduce memory usage ----')
ccm_obj = boto3.client('s3').get_object(Bucket=tableau_bucket, Key=ccm_key)
ccm = pd.read_csv(BytesIO(ccm_obj['Body'].read()),
                           sep=';',
                           quotechar='"',
                  index_col=0,
                           dtype = 'object')
reduce_mem_usage(ccm)
ccm['country'] = ccm['country'].str.upper()

print('\n ---- Add Champions with FULL ACCESS roles ----')
#Preprocessing champions frame

champions = champions[((champions['Department'].fillna('').str.contains(pat = 'DI'))&(champions['OrgUnit (SCD)'].fillna('').str.contains(pat = 'RC-')))|(champions['OrgUnit (SCD)']=='DI')]

champions['GID'] = champions['GID Technical'].str.upper()
del champions['GID Technical']
champions['USER_COUNTRY'] = champions['Country'].str.upper()
del champions['Country']
champions['ORIGINAL_ROLE'] = champions['Role'].str.upper()
champions['LEFT_ROLE'] = champions['Role'].str[:2]
champions['USER_ROLE'] = 'FULL ACCESS'
champions = champions[['GID', 'USER_ROLE', 'ORIGINAL_ROLE', 'LEFT_ROLE', 'USER_COUNTRY']]

champions = champions.merge(ccm, left_on='USER_COUNTRY', right_on='country', how='left')
if champions.COUNTRY_CODE.isna().sum() == 0:
    print('all champions countries mapped correctly')
    
new_champions_1 = champions.merge(champions_ifas, how = 'inner', left_on = 'LEFT_ROLE', right_on = 'Country-2char')
new_champions_2 = champions.merge(champions_ifas, how = 'inner', left_on = 'COUNTRY_CODE', right_on = 'Country-2char')

new_champions = pd.concat([new_champions_1, new_champions_2])
new_champions = new_champions.drop_duplicates()

del new_champions['LEFT_ROLE']
del new_champions['COUNTRY_CODE']

#delete already exisitng reps
if len(set(authorization['GID'].str.upper()).intersection(set(new_champions['GID'].str.upper()))) >0:
    print('users that are in HCT and are also champions', set(authorization['GID'].str.upper()).intersection(set(new_champions['GID'].str.upper())))
    authorization = authorization[~authorization['GID'].isin(set(authorization['GID'].str.upper()).intersection(set(new_champions['GID'].str.upper())))]

#add ifa info
new_champions = new_champions.merge(ifa_info, on='IFA', how='left')
new_champions = new_champions.merge(user_info, on='GID', how='left')
new_champions.columns


new_champions['ACCOUNT_COUNTRY'] = new_champions['ACCOUNT_COUNTRY'].fillna(new_champions['Country-2char'])
new_champions['ACCOUNT_NAME'] = new_champions['ACCOUNT_NAME'].fillna(new_champions['Account Name '])
new_champions['ACCOUNTID'] = new_champions['ACCOUNTID'].fillna(new_champions['Account ID'])
new_champions['TRANSPARENCY_POSITION'] = 'FULL ACCESS'
new_champions['USER_ID'] = 'FULL ACCESS'
new_champions = new_champions[['ACCOUNTID', 'IFA', 'COMPANY_ID', 'ACCOUNT_NAME', 'USER_ID',
                               'ACCOUNT_COUNTRY', 'GID', 'TRANSPARENCY_POSITION', 'USER_ROLE',
                               'USER_COUNTRY']]

authorization = pd.concat([authorization, new_champions])

print('\n ---- Create new columns IFA_GID & IFA_GID_ROLE ----')
authorization['IFA_GID'] = authorization['IFA'].astype(str) + '_' + authorization['GID'].astype(str)
authorization['IFA_GID_ROLE'] = authorization['IFA'].astype(str) + '_' + authorization['GID'].astype(str) + '_' + authorization['USER_ROLE'].astype(str)


print('\n ---- Map sales countries and Lead Countries ----')
lc_sc_mapping['SALES_COUNTRY'] = lc_sc_mapping.groupby(['LEAD_COUNTRY'])['SALES_COUNTRY'].transform(lambda x : ', '.join(str(v) for v in x)) 
lc_sc_mapping = lc_sc_mapping.drop_duplicates().reset_index(drop=True)
authorization['LEFT_USER_ROLE'] = authorization['USER_ROLE'].str[:2]
lc_sc_mapping['LEFT_LEAD_COUNTRY'] = lc_sc_mapping['LEAD_COUNTRY'].str[:2]

authorization = authorization.merge(lc_sc_mapping, left_on='LEFT_USER_ROLE', right_on='LEFT_LEAD_COUNTRY', how='left')

authorization['SALES_COUNTRY'] = np.where(authorization['USER_ROLE'].str.len()==8, authorization['USER_ROLE'].str[-2:], authorization['SALES_COUNTRY'])
authorization['SALES_COUNTRY'] = np.where(authorization['USER_ROLE'].str.len()>8, ' ', authorization['SALES_COUNTRY'])

print('\n ---- Map ifas with currency codes ----')
authorization = authorization.merge(curr_codes[['COUNTRY_CODE','CURR']],left_on='ACCOUNT_COUNTRY', right_on='COUNTRY_CODE', how='left')
print('countries with currency problems \n', authorization[authorization['CURR'].isna()==True]['ACCOUNT_COUNTRY'].value_counts())
authorization['CURR'] = np.where(authorization['ACCOUNT_COUNTRY']=='TW','TWD', authorization['CURR'] )
authorization['CURR'] = authorization['CURR'].fillna('EUR')


print('\n ---- Clean conflicting characters ----')
authorization = clean_conflicting_chars(authorization)

#Add new users to ifas
for index, row in add_user.iterrows():
    if len(authorization[(authorization['GID'] == row['GID'])&(authorization['IFA'] == row['IFA'].zfill(10))]) == 0: #if ifa and gid not featured before
        
        if len(authorization[authorization['GID'] == row['GID']])== 0: #if gid featured before
            new_user = pd.DataFrame({'USER_ID': [np.nan], 'GID': row['GID'], 'TRANSPARENCY_POSITION': ['FULL ACCESS'], 
                            'USER_ROLE': ['FULL ACCESS'], 'FULL ACCESS': ['FU'],'LEAD_COUNTRY': [np.nan],
                            'SALES_COUNTRY': [np.nan], 'LEFT_LEAD_COUNTRY': [np.nan] })
        else:
            new_user = authorization[authorization['GID']==row['GID']][['USER_ID', 'GID', 'TRANSPARENCY_POSITION','USER_ROLE','LEFT_USER_ROLE','LEAD_COUNTRY','SALES_COUNTRY','LEFT_LEAD_COUNTRY']].drop_duplicates().reset_index(drop=True)
        
        new_ifa = authorization[authorization['IFA']==row['IFA'].zfill(10)][['ACCOUNTID','IFA','COMPANY_ID','ACCOUNT_NAME','ACCOUNT_COUNTRY','CURR']].drop_duplicates().reset_index(drop=True)
        new_line = new_user.merge(new_ifa, how='left', left_index=True, right_index=True)

        new_line['IFA_GID'] = new_line['IFA'].astype(str) + '_' + new_line['GID'].astype(str)
        new_line['IFA_GID_ROLE'] = new_line['IFA'].astype(str) + '_' + new_line['GID'].astype(str) + '_' + new_line['USER_ROLE'].astype(str)
        authorization = pd.concat([authorization, new_line])

    else:
        print('No')
        continue
        
        
print('\n ---- Add Update Dates ----')
authorization['LAST_MODIFIED_DATE'] = ag_obj['LastModified']
authorization['LAST_PREPROCESSED_DATE'] = datetime.now()
print('Last modified date:', ag_obj['LastModified'])
print('Last preprocessed date:', authorization['LAST_PREPROCESSED_DATE'].max())



authorization['IFA'] = authorization['IFA'].str.zfill(10)
authorization['IFA'] = authorization['IFA'].astype(str)



print('\n ---- Save ifa_frame ----')
ifa_frame = authorization[['IFA','ACCOUNT_COUNTRY']].drop_duplicates()
ifa_frame = clean_conflicting_chars(ifa_frame)
ifa_frame.reset_index(drop=True, inplace=True)
csv_buffer = StringIO()
ifa_frame.to_csv(csv_buffer, index_label='INDEX', sep=';', index=False)
gzip_object = gzip.compress(csv_buffer.getvalue().encode('utf-8'))
obj = boto3.resource('s3').Object(tableau_bucket,
                                  'tableau-files/ifa_frame/ifa_gid_mapping.csv.gz')
obj.put(Body=gzip_object, ServerSideEncryption='aws:kms')


print('\n ---- Save ifa_gid_mapping ----')
IFA_GID = authorization[['IFA', 'GID']].drop_duplicates().sort_values(by='IFA')
IFA_GID.reset_index(drop=True, inplace=True)
csv_buffer = StringIO()
IFA_GID.to_csv(csv_buffer, index_label='INDEX', sep=';', index=False)
gzip_object = gzip.compress(csv_buffer.getvalue().encode('utf-8'))
obj = boto3.resource('s3').Object(tableau_bucket,
                                  'interm/ifa_gid_mapping.csv.gz')
obj.put(Body=gzip_object, ServerSideEncryption='aws:kms')



print('\n ---- Save ifa_accountid_mapping ----')
account_mapping = authorization[['ACCOUNTID', 'IFA']].drop_duplicates().sort_values(by='IFA')
csv_buffer = StringIO()
account_mapping.to_csv(csv_buffer, index_label='INDEX', sep=';', index=False)
gzip_object = gzip.compress(csv_buffer.getvalue().encode('utf-8'))
obj = boto3.resource('s3').Object(tableau_bucket, 
                                  'interm/ifa_accountid_mapping.csv.gz')
obj.put(Body=gzip_object, ServerSideEncryption='aws:kms')


print('\n ---- Save ifa_currency mapping----')
ifa_currency = authorization[['IFA','CURR','COUNTRY_CODE']].drop_duplicates().sort_values(by='IFA')
csv_buffer = StringIO()
ifa_currency.to_csv(csv_buffer, index_label='INDEX', sep=';', index=False)
gzip_object = gzip.compress(csv_buffer.getvalue().encode('utf-8'))
obj = boto3.resource('s3').Object(tableau_bucket, 
                                  'interm/ifa_currency.csv.gz')
obj.put(Body=gzip_object, ServerSideEncryption='aws:kms')


print('\n ---- Save account_name_mapping mapping----')
acc_name_map = account_grip[['IFA', 'ACCOUNT_NAME']].drop_duplicates().sort_values(by='IFA')
csv_buffer = StringIO()
acc_name_map.to_csv(csv_buffer, index_label='INDEX', sep=';', index=False)
gzip_object = gzip.compress(csv_buffer.getvalue().encode('utf-8'))
obj = boto3.resource('s3').Object(tableau_bucket, 
                                  'interm/account_name_mapping.csv.gz')
obj.put(Body=gzip_object, ServerSideEncryption='aws:kms')




authorization = authorization[['ACCOUNTID', 'IFA', 'COMPANY_ID', 'ACCOUNT_NAME', 'USER_ID',
                                   'ACCOUNT_COUNTRY', 'GID', 'TRANSPARENCY_POSITION', 'USER_ROLE',
                                   'IFA_GID', 'IFA_GID_ROLE', 'LEFT_USER_ROLE', 'LEAD_COUNTRY',
                                   'SALES_COUNTRY', 'LEFT_LEAD_COUNTRY', 'LAST_MODIFIED_DATE',
                                   'LAST_PREPROCESSED_DATE', 'CURR']]


# Divide dataframe into separate files of less than 10000 records each
chunks = np.array_split(authorization, np.ceil(len(authorization)/100000))


counter = 0
for chunk in chunks:
    counter = counter+1
    
    csv_buffer = StringIO()
    chunk.to_csv(csv_buffer, index_label='INDEX', sep=';', index=False)
    gzip_object = gzip.compress(csv_buffer.getvalue().encode('utf-8'))
    obj = boto3.resource('s3').Object(tableau_bucket,
        'tableau-files/authorization/Authorization_' + str(counter) +'.csv.gz')
    obj.put(Body=gzip_object, ServerSideEncryption='aws:kms')
    print(len(chunk))
print('File contains', len(set(authorization.IFA)), 'different IFAs and', len(set(authorization.GID)), 'different GIDs')

#Delete deprecated files
delete_deprecated_files(tableau_bucket, 'tableau-files/authorization/')

    
end_time = time()
time_taken = end_time - start_time
hours, rest = divmod(time_taken,3600)
minutes, seconds = divmod(rest, 60)
print('\n', int(hours), 'hours,', int(minutes), 'minutes and', int(seconds), 'seconds')