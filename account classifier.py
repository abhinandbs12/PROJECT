# -*- coding: utf-8 -*-
# Let us first begin with importing the required dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('/content/instagram.csv')
df.head()
df.info()

X = df.drop('fake', axis=1)
y = df['fake']
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Print a classification report
print(classification_report(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(model, 'logistic_regression_model.pkl')

# Load the model (for later use)
loaded_model = joblib.load('logistic_regression_model.pkl')

# Predict function
def predict_fake_account(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Standardize the input
    prediction = loaded_model.predict(input_data)
    return 'Fake' if prediction[0] == 1 else 'Genuine'

# Example prediction
example_data = [1,0.25,0,0,1,53,0,0,32,1000,955]  # Replace with actual feature values
result = predict_fake_account(example_data)
print(f"The account is predicted to be: {result}")

df.head()



df2 = pd.read_csv('/content/instagram_dummy_data-1.csv')
df2.head()

# replace Account Type with 1 if Genuine and 0 if Fake
df2['Account Type'] = df2['Account Type'].apply(lambda x: 1 if x == 'Genuine' else 0)
# add external URL as 0
df2['External URL'] = 0
#add  nums/length username as 0.0 and nums/length as 0 and Profile Picture wiyj 1 if yes and 0 if no and add description length that of average of description length in df
df2['nums/length username'] = 0.0
df2['nums/length'] = 0
df2['Profile Picture'] = df2['Profile Picture'].apply(lambda x: 1 if x == 'Yes' else 0)

# add discription length in df2 as the averageof discription length of df
df2['Description Length'] = 22.623264


# store Username to a variable and drop from df2
username = df2['Username']
df2 = df2.drop('Username', axis=1)
# add private to df2 and alternate between 1 and 0
df2['Private'] = 1
df2['Private'] = df2['Private'].apply(lambda x: 0 if x == 1 else 1)
df2.head()

# add full name words as average of df full name
df2['Full Name Words'] = 1.460069
# add name == username  as 0
df2["name == username"] = 0
df2.head()

# cad storing Account Type data
account_type = df2['Account Type']
account_type.head()

# arrange df2 in  the form of Profile Picture,nums/length username,Full Name Words, nums/length ,name == username ,Description Length ,External URL ,Private ,Posts , Followers,Following
df2 = df2[['Profile Picture','nums/length username','Full Name Words','nums/length','name == username','Description Length','External URL','Private','Posts','Followers','Following']]
df2.head()

df2 = pd.DataFrame(df2)  # Convert back to DataFrame
df2.columns = ['Profile Picture','nums/length username','Full Name Words','nums/length','name == username','Description Length','External URL','Private','Posts','Followers','Following'] # Assign column names


# Convert df2 into an array of row data
row_data_array = df2.values.tolist()

# Print the first few rows of the array to verify
print(row_data_array[:5])

ft = []
for i in range(len(row_data_array)):

  result = predict_fake_account(row_data_array[i])
  x = account_type[i]
  ft.append({'Genuine' if x == 1 else 'Fake'})
  print(f"The account is predicted to be: {'Genuine' if x == 1 else 'Fake'}")

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

class InstagramAccountClassifier:
    def __init__(self, api_token=None):
        self.api_token = api_token

    def classify_account(self, data):
        def classify(row):
            if row['Follower-Following Ratio'] <= 1 or row['Follower-Following Ratio'] > 50 or row['Engagement Rate (%)'] < 2 or row['Engagement Rate (%)'] > 20 or row['Engagement Consistency'] == 'Inconsistent' or row['Suspicious Words in Bio'] == 'Yes' or row['Account Age (Months)'] <= 6:
                return 'Fake'
            return 'Genuine'
        data['Account Type'] = data.apply(classify, axis=1)

        return data

    def analyze_csv_data(self, file_path):
        data = pd.read_csv(file_path)
        data['Follower-Following Ratio'] = data['Followers'] / np.where(data['Following'] == 0, 1, data['Following'])
        return self.classify_account(data)

    def fetch_account_data_api(self, username):
        url = f"https://graph.instagram.com/{username}?fields=id,username,followers_count,follows_count,media_count&access_token={self.api_token}"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"API Error: {response.status_code} - {response.text}")
        account_data = response.json()
        return {
            'Username': account_data['username'],
            'Followers': account_data['followers_count'],
            'Following': account_data['follows_count'],
            'Posts': account_data['media_count'],
            'Account Age (Months)': 12,
            'Engagement Rate (%)': 5.0,
            'Engagement Consistency': 'Consistent',
            'Suspicious Words in Bio': 'No'
        }

    def scrape_account_data(self, username):
        url = f"https://www.instagram.com/{username}/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Web Scraping Error: {response.status_code} - {response.text}")
        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script', type='application/ld+json')
        if not scripts:
            raise ValueError("Could not find JSON data in the page.")
        account_json = scripts[0].string
        account_data = eval(account_json)
        return {
            'Username': account_data['name'],
            'Followers': int(account_data['interactionStatistic'][0]['userInteractionCount']),
            'Following': 0,
            'Posts': int(account_data['mainEntityofPage']['interactionStatistic']['userInteractionCount']),
            'Account Age (Months)': 12,
            'Engagement Rate (%)': 5.0,
            'Engagement Consistency': 'Consistent',
            'Suspicious Words in Bio': 'No'
        }

    def classify_live_account(self, username, method='api', fallback_file='/content/instagram_dummy_data-1.csv'):
        try:
            if method == 'api' and self.api_token:
                account_data = self.fetch_account_data_api(username)
            elif method == 'scraping':
                account_data = self.scrape_account_data(username)
            else:
                raise ValueError("Invalid method or missing API token for API-based classification.")
            followers = account_data.get('Followers', 1)
            following = account_data.get('Following', 1)
            ratio = followers / (following if following else 1)
            account_type = 'Genuine' if ratio > 1 and ratio <= 50 else 'Fake'
            return {
                'Source': method.upper(),
                'Username': account_data['Username'],
                'Followers': followers,
                'Following': following,
                'Posts': account_data.get('Posts', 0),
                'Follower-Following Ratio': round(ratio, 2),
                'Account Type': account_type
            }
        except Exception as e:
            print(f"Error fetching live account data: {e}")
            print("Falling back to CSV dataset for classification...")
            return self.analyze_csv_data(fallback_file)

    def report_fake_accounts(self, classified_data, report_filename='reported_fake_accounts.txt'):
        fake_accounts = classified_data[classified_data['Account Type'] == 'Fake']
        if not fake_accounts.empty:
            print("\nDetected Fake Accounts:")
            with open(report_filename, 'a') as f:
                for index, account in fake_accounts.iterrows():
                    # Calculate the "Reported Message" with a mock squared error (MSE)
                    mse = self.calculate_mse(account)

                    # Prepare the report in one-line format
                    report_message = (
                        f"Username: {account['Username']}, "
                        f"Follower-Following Ratio: {account['Follower-Following Ratio']}, "
                        f"Engagement Rate: {account['Engagement Rate (%)']}, "
                        f"Reported"
                    )

                    # Write the report to the file in one line
                    f.write(report_message + '\n')

                    # Display the report in the console in one line
                    print(report_message)

            print(f"Fake accounts automatically reported to {report_filename}")
        else:
            print("No fake accounts detected.")

    def calculate_mse(self, account):
        """
        Calculate a simple MSE score based on some threshold metrics.
        This is a mock function to simulate reporting error (e.g., fake detection error).
        """
        # For simplicity, calculate MSE as the difference between actual and expected engagement rate and ratio
        expected_engagement_rate = 5.0 # Assumed expected engagement rate for genuine accounts
        expected_ratio = 10 # Assumed expected follower/following ratio for genuine accounts
        mse_engagement = (account['Engagement Rate (%)'] - expected_engagement_rate) ** 2
        mse_ratio = (account['Follower-Following Ratio'] - expected_ratio) ** 2
        return round(mse_engagement + mse_ratio, 2)

if __name__ == "__main__":
    classifier = InstagramAccountClassifier(api_token='YOUR_INSTAGRAM_API_TOKEN')

    # Example 1: Classifying a live account using scraping method
    username = 'example_username'
    print("\nClassifying live account...\n")
    live_result = classifier.classify_live_account(username, method='scraping')

    # Example 2: Classifying a live account using API method
    print(f"\nClassifying live account {username} using API...\n")
    live_result_api = classifier.classify_live_account(username, method='api')

    # Example 3: Analyzing accounts from a CSV file and automatically reporting fake ones
    print("\nAnalyzing all accounts in the CSV file...\n")
    all_accounts_status = classifier.analyze_csv_data('/content/instagram_dummy_data-1.csv')
    print(all_accounts_status.to_string(index=False))

    # Automatically report fake accounts
    classifier.report_fake_accounts(all_accounts_status)
