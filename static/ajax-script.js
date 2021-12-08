$(document).ready(function () {
    $(".filter_button").click(function (e) {
        e.preventDefault();
        $.ajax({
            type: "POST",
            url: "tasks",
            data: {
                avatar: avatar,
            },
            error: function (result) {
                alert('error');
            }
        });
    });

    $(".task_button").click(function (e) {
        e.preventDefault();
        console.log(e)
        $.ajax({
            type: "POST",
            url: "tasks",
            data: {
                animate: 1,
            },
            error: function (result) {
                alert('error');
            }
        });
    });
});

const buttons = document.querySelectorAll('.filter_button');
let avatar;

buttons.forEach(elem => {
    elem.addEventListener("click", (e) => {
        avatar = elem.childNodes[1].attributes.src.value;

        buttons.forEach(function (e) {
            e.classList.remove('active');
        });
        elem.classList.add('active')
    })

})
