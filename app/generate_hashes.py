# generate_hashes.py  – run once on your laptop / REPL
import streamlit_authenticator as stauth
usernames   = ["alice", "bob"]
plain_pw    = ["4l1c3_P@ss", "B0B_P4ssw0rd"]

hasher = stauth.Hasher()
hashed_passwords = stauth.Hasher(plain_pw).generate()


for h in hashed_passwords:
    print(h)