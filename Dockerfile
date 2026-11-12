# Railway 部署：以 nginx 提供 dist/ 內的兩個靜態 HTML（計算都在使用者瀏覽器內，無後端）
FROM nginx:1.27-alpine

# Railway 會注入 PORT；本機預設 8080
ENV PORT=8080
COPY deploy/default.conf.template /etc/nginx/templates/default.conf.template

COPY deploy/index.html /usr/share/nginx/html/index.html
COPY dist/ /usr/share/nginx/html/
# 另存英文檔名，網址較好分享
COPY ["dist/撥貸金額預測工具.html", "/usr/share/nginx/html/predictor.html"]
COPY ["dist/撥貸金額(決策樹判斷).html", "/usr/share/nginx/html/tree.html"]

EXPOSE 8080
