class Solution:
    def shortestPalindrome(self, s: str) -> str:
        j = 0
        for i in range(len(s),-1,-1):
            if s[:i] == s[:i][::-1]:
                app = s[i:][::-1]
                return app + s[j:]
            else:
                j += 0

